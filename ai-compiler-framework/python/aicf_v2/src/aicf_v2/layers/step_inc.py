from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.step_inc import step_inc as emit_step_inc


class StepInc(Layer):
    """
    Step increment kernel wrapper.

    Contract:
      inputs : [step]    (i32 scalar; v2 uses shape=(1,))
      outputs: [step_out]
    """

    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, step: int, *, ctx: CudaEmitContext) -> int:
        ss = b.values[step].spec
        if ss.dtype != "i32":
            raise ValueError(f"StepInc expects step dtype i32; got {ss.dtype}")
        if tuple(ss.shape) != (1,):
            raise ValueError(f"StepInc expects step shape (1,) in v2; got {ss.shape}")

        out_step = b.value(f"{self.name}.out", TensorSpec(shape=(1,), dtype="i32", device=ss.device))

        emit_step_inc(
            b, ctx,
            step=step,
            out_step=out_step,
            name=f"{self.name}.step_inc",
            constraints={"inplace_ok": True},
        )
        return out_step
