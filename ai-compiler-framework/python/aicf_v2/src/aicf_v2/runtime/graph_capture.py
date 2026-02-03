from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from ..model import Model
from ..builder import Builder
from ..compile.types import CompiledProgram
from ..graph import Op
from ..backends.cuda.bridge import op_call, current_stream_u64
from .alloc import alloc_from_spec, _device_ok, _torch_dtype  # reuse existing helpers


# -----------------------------
# helpers
# -----------------------------
def _check_feed_tensor(name: str, spec, t: torch.Tensor) -> torch.Tensor:
    if tuple(t.shape) != tuple(spec.shape):
        raise ValueError(f"Feed shape mismatch for '{name}': got {tuple(t.shape)} expected {spec.shape}")

    if not _device_ok(spec.device, t):
        raise ValueError(f"Feed device mismatch for '{name}': got {t.device} expected {spec.device}")

    exp = _torch_dtype(spec.dtype)
    if t.dtype != exp:
        raise ValueError(f"Feed dtype mismatch for '{name}': got {t.dtype} expected {exp}")

    if not t.is_contiguous():
        t = t.contiguous()

    return t


def _role_vids(b: Builder, role: str) -> List[int]:
    if role == "input":
        return list(getattr(b, "input_vids", []))
    if role == "param":
        return list(getattr(b, "param_vids", []))
    if role == "state":
        return list(getattr(b, "state_vids", []))
    raise KeyError(f"unknown role '{role}'")


def _all_external_vids(b: Builder) -> List[int]:
    return list(getattr(b, "external_vids", []))


def _apply_abi_fixups(op: Op, ins: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Same policy as runtime/cuda_exec.py:
      hints['view_rank0_inputs'] = [idx...]
    """
    hints = getattr(op, "hints", None) or {}
    idxs = hints.get("view_rank0_inputs", None)
    if idxs:
        for i in idxs:
            if 0 <= i < len(ins):
                t = ins[i]
                if t.dim() == 1 and t.numel() == 1:
                    ins[i] = t.view(())
    return ins


# -----------------------------
# captured program container
# -----------------------------
@dataclass
class GraphCapturedProgram:
    prog: CompiledProgram
    slots: List[torch.Tensor]                     # slots[vid] -> Tensor
    ext_bufs: Dict[int, torch.Tensor]             # external vid -> fixed buffer tensor
    graph: torch.cuda.CUDAGraph
    static_roles: Tuple[str, ...]
    copy_roles: Tuple[str, ...]


# -----------------------------
# core capture / replay
# -----------------------------
def capture_cuda_graph(
    m: Model,
    prog: CompiledProgram,
    feed: Dict[str, torch.Tensor],
    *,
    stream: Optional[int] = None,
    warmup: int = 2,
    static_roles: Iterable[str] = ("input", "param"),
    copy_roles: Iterable[str] = ("input",),
) -> GraphCapturedProgram:
    """
    Capture CUDA Graph with fixed pointer policy.

    Design:
      - all externals must be backed by fixed buffers (ext_bufs) so pointers don't change
      - replay copies only roles in copy_roles into those buffers
      - alias decisions applied before running ops

    NOTE:
      - capture occurs on CURRENT torch stream.
      - 'stream' (u64) is used only when calling op_call.
    """
    if stream is not None:
        raise NotImplementedError("capture_cuda_graph(stream=...) not supported yet; capture uses current torch stream")

    b: Builder = m.b
    plan = prog.plan
    ops = plan.ops

    static_roles = tuple(static_roles)
    copy_roles = tuple(copy_roles)

    # 1) externals must be covered by static_roles, otherwise pointers could change
    static_vids: List[int] = []
    for r in static_roles:
        static_vids += _role_vids(b, r)
    static_set = set(static_vids)

    ext_vids = _all_external_vids(b)
    missing = [vid for vid in ext_vids if vid not in static_set]
    if missing:
        names = [b.values[v].name for v in missing]
        raise ValueError(
            f"CUDA Graph requires fixed pointers for all externals. "
            f"Missing from static_roles: {names}  static_roles={static_roles}"
        )

    # 2) allocate slots
    #    - externals: allocate fixed buffers and copy feed into them once
    #    - internals: allocate by spec
    slots: List[Optional[torch.Tensor]] = [None] * len(b.values)
    ext_bufs: Dict[int, torch.Tensor] = {}

    for vid in ext_vids:
        v = b.values[vid]
        name = v.name
        if name not in feed:
            raise KeyError(f"Missing feed for external '{name}'")
        src = _check_feed_tensor(name, v.spec, feed[name])

        buf = alloc_from_spec(v.spec)     # fixed address for graph lifetime
        buf.copy_(src)
        slots[vid] = buf
        ext_bufs[vid] = buf

    for v in b.values:
        if slots[v.vid] is None:
            slots[v.vid] = alloc_from_spec(v.spec)

    slots_t: List[torch.Tensor] = [t for t in slots]  # type: ignore

    # 3) apply alias decisions (inplace)
    for out_vid, in_vid in plan.alias.items():
        slots_t[out_vid] = slots_t[in_vid]
        if out_vid in ext_bufs:
            ext_bufs[out_vid] = slots_t[in_vid]

    # 4) warmup (eager launches, same buffers)
    stream_u64 = current_stream_u64()
    for _ in range(int(warmup)):
        _copy_roles_into_ext_bufs(b, ext_bufs, feed, copy_roles=copy_roles)
        _run_ops(ops, slots_t, stream_u64)

    torch.cuda.synchronize()

    # 5) capture
    graph = torch.cuda.CUDAGraph()
    _copy_roles_into_ext_bufs(b, ext_bufs, feed, copy_roles=copy_roles)
    torch.cuda.synchronize()

    with torch.cuda.graph(graph):
        _run_ops(ops, slots_t, stream_u64)

    torch.cuda.synchronize()

    return GraphCapturedProgram(
        prog=prog,
        slots=slots_t,
        ext_bufs=ext_bufs,
        graph=graph,
        static_roles=static_roles,
        copy_roles=copy_roles,
    )


def replay_cuda_graph(
    m: Model,
    gprog: GraphCapturedProgram,
    feed: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Replay captured graph.

    Policy:
      - copy feed tensors for roles in gprog.copy_roles into ext_bufs
      - graph.replay()
      - return outputs
    """
    b: Builder = m.b

    _copy_roles_into_ext_bufs(b, gprog.ext_bufs, feed, copy_roles=gprog.copy_roles)
    gprog.graph.replay()

    out: Dict[str, torch.Tensor] = {}
    if getattr(b, "outputs", None):
        for oname, vid in b.outputs.items():
            out[oname] = gprog.slots[vid]
    else:
        for vid in b.output_vids:
            out[b.values[vid].name] = gprog.slots[vid]
    return out


# -----------------------------
# internal ops
# -----------------------------
def _copy_roles_into_ext_bufs(
    b: Builder,
    ext_bufs: Dict[int, torch.Tensor],
    feed: Dict[str, torch.Tensor],
    *,
    copy_roles: Tuple[str, ...],
) -> None:
    copy_set = set(copy_roles)
    for role in ("input", "param", "state"):
        if role not in copy_set:
            continue
        for vid in _role_vids(b, role):
            v = b.values[vid]
            name = v.name
            if name not in feed:
                raise KeyError(f"Missing feed for '{name}' (role={role})")
            src = _check_feed_tensor(name, v.spec, feed[name])

            if vid not in ext_bufs:
                raise KeyError(f"internal error: ext buffer missing for vid={vid} name='{name}'")
            ext_bufs[vid].copy_(src)


def _run_ops(ops: List[Op], slots: List[torch.Tensor], stream_u64: int) -> None:
    for op in ops:
        # emitter must fill caches
        if getattr(op, "kind_id", None) is None:
            raise ValueError(f"[graph_capture] missing op.kind_id (kind='{op.kind}', name='{op.name}')")
        if getattr(op, "attr_schema", None) is None:
            raise ValueError(f"[graph_capture] missing op.attr_schema (kind='{op.kind}', name='{op.name}')")
        if getattr(op, "attr_blob", None) is None:
            raise ValueError(f"[graph_capture] missing op.attr_blob (kind='{op.kind}', name='{op.name}')")

        ins = [slots[v] for v in op.inputs]
        outs = [slots[v] for v in op.outputs]

        # ABI fixups by hints
        ins = _apply_abi_fixups(op, ins)

        op_call(
            int(op.kind_id),
            ins, outs,
            int(op.attr_schema),
            op.attr_blob,
            stream=stream_u64,
        )
