from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple

import torch

from ..model import Model
from ..builder import Builder
from ..compile.types import CompiledProgram
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
      - alias decisions applied before running lowered ops

    NOTE:
      - capture occurs on CURRENT torch stream.
      - 'stream' (u64) is used only when calling op_call; if you want non-default
        torch stream capture, wire torch.cuda.Stream into executor first.
    """
    if stream is not None:
        # 현재 구조에서 torch.cuda.CUDAGraph 캡처 스트림과 u64 stream 핸들을
        # 1:1로 맞추는 배선이 없어서 여기선 막아두는 게 안전함.
        raise NotImplementedError("capture_cuda_graph(stream=...) not supported yet; capture uses current torch stream")

    b: Builder = m.b
    plan = prog.plan
    lowered = plan.lowered

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

    # cast Optional away
    slots_t: List[torch.Tensor] = [t for t in slots]  # type: ignore

    # 3) apply alias decisions (inplace)
    for out_vid, in_vid in plan.alias.items():
        slots_t[out_vid] = slots_t[in_vid]
        # keep ext_bufs mapping consistent if out_vid is external (rare but possible)
        if out_vid in ext_bufs:
            ext_bufs[out_vid] = slots_t[in_vid]

    # 4) warmup (eager launches, same buffers)
    stream_u64 = current_stream_u64()
    for _ in range(int(warmup)):
        _copy_roles_into_ext_bufs(b, ext_bufs, feed, copy_roles=copy_roles)
        _run_lowered_ops(lowered, slots_t, stream_u64)

    torch.cuda.synchronize()

    # 5) capture
    graph = torch.cuda.CUDAGraph()
    _copy_roles_into_ext_bufs(b, ext_bufs, feed, copy_roles=copy_roles)
    torch.cuda.synchronize()

    with torch.cuda.graph(graph):
        _run_lowered_ops(lowered, slots_t, stream_u64)

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
    # copy only selected roles (train: inputs only; inference: inputs only by default)
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
                # should not happen: all externals must have buffers
                raise KeyError(f"internal error: ext buffer missing for vid={vid} name='{name}'")
            ext_bufs[vid].copy_(src)


def _run_lowered_ops(lowered, slots: List[torch.Tensor], stream_u64: int) -> None:
    for lop in lowered:
        ins = [slots[v] for v in lop.in_vids]
        outs = [slots[v] for v in lop.out_vids]

        # FIX: ABI shim (adam_step expects rank0 scalars for bc1/bc2 in current C++ impl)
        if lop.kind == "adam_step":
            if len(ins) >= 6:
                bc1 = ins[4]
                bc2 = ins[5]
                if bc1.dim() == 1 and bc1.numel() == 1:
                    ins[4] = bc1.view(())
                if bc2.dim() == 1 and bc2.numel() == 1:
                    ins[5] = bc2.view(())

        op_call(
            lop.kind_id,
            ins, outs,
            lop.attr_schema,
            lop.attr_blob,
            stream=stream_u64,
        )
