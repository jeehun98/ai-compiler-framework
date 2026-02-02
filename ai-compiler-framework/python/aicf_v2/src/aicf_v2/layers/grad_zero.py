from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.grad_zero import grad_zero as emit_grad_zero


class GradZero(Layer):
    """
    Zero out gradient buffer.

    Contract:
      inputs : [x]
      outputs: [y]   (planner may alias/inplace later)
    """

    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        xs = b.values[x].spec

        # keep as separate value; plan may alias if inplace is allowed
        y = b.value(f"{self.name}.out", TensorSpec(shape=xs.shape, dtype=xs.dtype, device=xs.device))

        emit_grad_zero(
            b, ctx,
            x=x,
            out=y,
            name=f"{self.name}.grad_zero",
            constraints={"inplace_ok": True},
        )
        return y
