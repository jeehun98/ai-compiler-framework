from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch

from ..model import Model
from ..builder import Builder
from ..compile.compile import compile_cuda
from ..compile.types import CompiledProgram, LoweredOp
from ..backends.cuda.registry import CudaRegistry
from ..backends.cuda.bridge import op_call, current_stream_u64
from .alloc import bind_and_alloc_slots
from .graph_capture import GraphCapturedProgram, capture_cuda_graph, replay_cuda_graph


def _tensor_sig(t: torch.Tensor) -> Tuple[Tuple[int, ...], str, str]:
    return (tuple(t.shape), str(t.dtype), str(t.device))


def _feed_signature(b: Builder, feed: Dict[str, torch.Tensor]) -> Tuple[Tuple[str, Tuple[Tuple[int, ...], str, str]], ...]:
    items = []
    for vid in getattr(b, "external_vids", []):
        name = b.values[vid].name
        if name not in feed:
            raise KeyError(f"Missing feed for external '{name}'")
        items.append((name, _tensor_sig(feed[name])))
    items.sort(key=lambda x: x[0])
    return tuple(items)


def _apply_abi_fixups(lop: LoweredOp, ins: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Apply ABI fixups using per-op hints (emitter-provided).

    Supported hints:
      - view_rank0_inputs: list[int]
          For each input index i, if ins[i] is rank1 scalar (shape (1,), numel==1),
          present it to the backend as rank0 via view(()) without copy.
    """
    hints = getattr(lop, "hints", None) or {}
    idxs = hints.get("view_rank0_inputs", None)
    if idxs:
        for i in idxs:
            if 0 <= i < len(ins):
                t = ins[i]
                if t.dim() == 1 and t.numel() == 1:
                    ins[i] = t.view(())
    return ins


class CudaExecutor:
    def __init__(self, registry: Optional[CudaRegistry] = None):
        self.registry = registry or CudaRegistry()
        self._compiled_cache: Dict[int, CompiledProgram] = {}
        self._graph_cache: Dict[Any, GraphCapturedProgram] = {}

    def compile(self, m: Model) -> CompiledProgram:
        return compile_cuda(m, self.registry)

    def compile_cached(self, m: Model) -> CompiledProgram:
        key = id(m.b)
        prog = self._compiled_cache.get(key)
        if prog is None:
            prog = self.compile(m)
            self._compiled_cache[key] = prog
        return prog

    def clear_cache(self) -> None:
        self._compiled_cache.clear()
        self._graph_cache.clear()

    def run(
        self,
        m: Model,
        feed: Dict[str, torch.Tensor],
        *,
        stream: Optional[int] = None,
        use_cuda_graph: bool = True,
        mode: str = "inference",
        warmup: int = 2,
    ) -> Dict[str, torch.Tensor]:
        prog = self.compile_cached(m)
        return self.run_compiled(
            m,
            prog,
            feed,
            stream=stream,
            use_cuda_graph=use_cuda_graph,
            mode=mode,
            warmup=warmup,
        )

    def _cache_key(
        self,
        m: Model,
        prog: CompiledProgram,
        feed: Dict[str, torch.Tensor],
        *,
        mode: str,
        static_roles: Tuple[str, ...],
        copy_roles: Tuple[str, ...],
    ):
        b = m.b
        plan_key = getattr(prog.plan, "plan_id", None) or id(prog.plan)
        return (mode, plan_key, static_roles, copy_roles, _feed_signature(b, feed))

    def run_compiled(
        self,
        m: Model,
        prog: CompiledProgram,
        feed: Dict[str, torch.Tensor],
        *,
        stream: Optional[int] = None,
        use_cuda_graph: bool = True,
        mode: str = "inference",
        warmup: int = 2,
    ) -> Dict[str, torch.Tensor]:
        b: Builder = m.b
        plan = prog.plan

        # -------------------------
        # Eager (no cuda graph)
        # -------------------------
        if not use_cuda_graph:
            stream_u64 = int(stream) if stream is not None else current_stream_u64()

            slots = bind_and_alloc_slots(b, feed)

            # apply alias decisions (inplace)
            for out_vid, in_vid in plan.alias.items():
                slots[out_vid] = slots[in_vid]

            for lop in plan.lowered:
                ins = [slots[v] for v in lop.in_vids]
                outs = [slots[v] for v in lop.out_vids]

                # ✅ ABI fixups via hints (no kind hardcode)
                ins = _apply_abi_fixups(lop, ins)

                op_call(
                    lop.kind_id,
                    ins, outs,
                    lop.attr_schema,
                    lop.attr_blob,
                    stream=stream_u64,
                )

            out: Dict[str, torch.Tensor] = {}
            if getattr(b, "outputs", None):
                for oname, vid in b.outputs.items():
                    out[oname] = slots[vid]
            else:
                for vid in b.output_vids:
                    out[b.values[vid].name] = slots[vid]
            return out

        # -------------------------
        # CUDA Graph path
        # -------------------------
        if stream is not None:
            raise NotImplementedError("use_cuda_graph=True with explicit stream is not supported yet")

        if mode == "train":
            static_roles = ("input", "param", "state")
            copy_roles = ("input",)
        else:
            static_roles = ("input", "param")
            copy_roles = ("input",)

        key = self._cache_key(
            m, prog, feed,
            mode=mode,
            static_roles=static_roles,
            copy_roles=copy_roles,
        )

        gprog = self._graph_cache.get(key)
        if gprog is None:
            gprog = capture_cuda_graph(
                m,
                prog,
                feed,
                stream=None,
                warmup=warmup,
                static_roles=static_roles,
                copy_roles=copy_roles,
            )
            self._graph_cache[key] = gprog

        return replay_cuda_graph(m, gprog, feed)
