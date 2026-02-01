from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import torch

from ..model import Model
from ..builder import Builder
from ..compile.compile import compile_cuda
from ..compile.types import CompiledProgram
from ..backends.cuda.registry import CudaRegistry
from ..backends.cuda.bridge import op_call, current_stream_u64
from .alloc import bind_and_alloc_slots
from .graph_capture import GraphCapturedProgram, capture_cuda_graph, replay_cuda_graph


def _tensor_sig(t: torch.Tensor) -> Tuple[Tuple[int, ...], str, str]:
    # (shape, dtype, device)
    # NOTE: do NOT include contiguity in cache key. We already contig() at bind/copy stage.
    return (tuple(t.shape), str(t.dtype), str(t.device))


def _feed_signature(b: Builder, feed: Dict[str, torch.Tensor]) -> Tuple[Tuple[str, Tuple[Tuple[int, ...], str, str]], ...]:
    """
    Stable signature used for CUDA graph cache.
    Uses ONLY externals and only shape/dtype/device.
    """
    items = []
    for vid in getattr(b, "external_vids", []):
        name = b.values[vid].name
        # FIX: make missing external a hard error (safer cache key & capture behavior)
        if name not in feed:
            raise KeyError(f"Missing feed for external '{name}'")
        items.append((name, _tensor_sig(feed[name])))
    items.sort(key=lambda x: x[0])
    return tuple(items)


def _fixup_inputs_for_backend(kind: str, ins: list[torch.Tensor]) -> list[torch.Tensor]:
    """
    Backend ABI fixups (temporary shims).

    - adam_step: C++ kernel test uses rank0 scalars for bc1/bc2.
      In v2 we often represent scalars as shape (1,).
      Use view(()) to present rank0 to the kernel without copying.
    """
    if kind == "adam_step":
        # inputs: [P, G, M, V, bc1, bc2]
        if len(ins) >= 6:
            bc1 = ins[4]
            bc2 = ins[5]
            if bc1.dim() == 1 and bc1.numel() == 1:
                ins[4] = bc1.view(())
            if bc2.dim() == 1 and bc2.numel() == 1:
                ins[5] = bc2.view(())
    return ins


class CudaExecutor:
    """
    Executor:
      - compile(Model) -> CompiledProgram(plan)
      - run_compiled(Model, CompiledProgram, feed) -> outputs
      - run(Model, feed) -> uses compiled cache for stable plan identity

    + optional CUDA Graph capture/replay cache
    """

    def __init__(self, registry: Optional[CudaRegistry] = None):
        self.registry = registry or CudaRegistry()

        # compiled cache: prevents plan identity changing every run()
        self._compiled_cache: Dict[int, CompiledProgram] = {}

        # cuda graph cache: key -> GraphCapturedProgram
        self._graph_cache: Dict[Any, GraphCapturedProgram] = {}

    # -------------------------
    # Compile
    # -------------------------
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

    # -------------------------
    # Public run
    # -------------------------
    def run(
        self,
        m: Model,
        feed: Dict[str, torch.Tensor],
        *,
        stream: Optional[int] = None,
        use_cuda_graph: bool = True,
        mode: str = "inference",  # "inference" | "train"
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

    # -------------------------
    # Graph cache key
    # -------------------------
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

    # -------------------------
    # Core execution
    # -------------------------
    def run_compiled(
        self,
        m: Model,
        prog: CompiledProgram,
        feed: Dict[str, torch.Tensor],
        *,
        stream: Optional[int] = None,
        use_cuda_graph: bool = True,
        mode: str = "inference",  # "inference" | "train"
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

            # execute lowered ops
            for lop in plan.lowered:
                ins = [slots[v] for v in lop.in_vids]
                outs = [slots[v] for v in lop.out_vids]

                # FIX: ABI shim (adam_step bc1/bc2 rank fix)
                ins = _fixup_inputs_for_backend(lop.kind, ins)

                op_call(
                    lop.kind_id,
                    ins, outs,
                    lop.attr_schema,
                    lop.attr_blob,
                    stream=stream_u64,
                )

            # outputs
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
        # FIX: be explicit (capture_cuda_graph currently rejects stream!=None)
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
