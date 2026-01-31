from __future__ import annotations
from typing import Dict, Optional

import torch

from ..model import Model
from ..backends.cuda.registry import CudaRegistry
from ..backends.cuda.bridge import op_call, current_stream_u64
from ..compiler.compile import compile_cuda
from ..compiler.types import CompiledProgram
from .alloc import bind_and_alloc_slots


class CudaExecutor:
    """
    Executor is now two-phase:
      - compile(Model) -> CompiledProgram(plan)
      - run_compiled(Model, CompiledProgram, feed) -> outputs
    """

    def __init__(self, registry: Optional[CudaRegistry] = None):
        self.registry = registry or CudaRegistry()

    def compile(self, m: Model) -> CompiledProgram:
        return compile_cuda(m, self.registry)

    def run(self, m: Model, feed: Dict[str, torch.Tensor], *, stream: Optional[int] = None) -> Dict[str, torch.Tensor]:
        prog = self.compile(m)
        return self.run_compiled(m, prog, feed, stream=stream)

    def run_compiled(
        self,
        m: Model,
        prog: CompiledProgram,
        feed: Dict[str, torch.Tensor],
        *,
        stream: Optional[int] = None
    ) -> Dict[str, torch.Tensor]:
        b = m.b
        plan = prog.plan
        stream_u64 = int(stream) if stream is not None else current_stream_u64()

        # 1) bind feed + alloc all slots
        slots = bind_and_alloc_slots(b, feed)

        # 2) apply alias (inplace decisions)
        for out_vid, in_vid in plan.alias.items():
            slots[out_vid] = slots[in_vid]

        # 3) execute lowered ops
        for lop in plan.lowered:
            ins = [slots[v] for v in lop.in_vids]
            outs = [slots[v] for v in lop.out_vids]

            # sanity
            assert all(isinstance(t, torch.Tensor) for t in ins)
            assert all(isinstance(t, torch.Tensor) for t in outs)

            op_call(
                lop.kind_id,
                ins, outs,
                lop.attr_schema,
                lop.attr_blob,
                stream=stream_u64,
            )

        # 4) outputs
        out: Dict[str, torch.Tensor] = {}

        # prefer explicit output mapping if present
        if hasattr(b, "outputs") and isinstance(getattr(b, "outputs"), dict) and b.outputs:
            for oname, vid in b.outputs.items():
                out[oname] = slots[vid]  # type: ignore
        else:
            for vid in b.output_vids:
                out[b.values[vid].name] = slots[vid]  # type: ignore

        return out