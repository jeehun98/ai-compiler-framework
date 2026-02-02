from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.copy import copy as emit_copy


class Copy(Layer):
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        x_spec = b.values[x].spec
        y = b.value(f"{self.name}.out", TensorSpec(shape=x_spec.shape, dtype=x_spec.dtype, device=x_spec.device))

        emit_copy(
            b, ctx,
            x=x,
            out=y,
            name=f"{self.name}.copy",
        )
        return y
