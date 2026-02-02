from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.relu_bwd import relu_bwd as emit_relu_bwd


class ReLUBwd(Layer):
    """
    ReLU backward.

    Contract:
      inputs : [dy, y]
      outputs: [dx]   (same spec as dy)
    """

    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, dy: int, y: int, *, ctx: CudaEmitContext) -> int:
        dy_spec = b.values[dy].spec
        y_spec = b.values[y].spec

        if tuple(y_spec.shape) != tuple(dy_spec.shape):
            raise ValueError(f"ReLUBwd shape mismatch: dy={dy_spec.shape} y={y_spec.shape}")
        if y_spec.dtype != dy_spec.dtype:
            raise ValueError(f"ReLUBwd dtype mismatch: dy={dy_spec.dtype} y={y_spec.dtype}")
        if y_spec.device != dy_spec.device:
            raise ValueError(f"ReLUBwd device mismatch: dy={dy_spec.device} y={y_spec.device}")

        dx = b.value(f"{self.name}.dx", TensorSpec(shape=dy_spec.shape, dtype=dy_spec.dtype, device=dy_spec.device))

        emit_relu_bwd(
            b, ctx,
            dy=dy,
            y=y,
            out_dx=dx,
            name=f"{self.name}.relu_bwd",
        )
        return dx
