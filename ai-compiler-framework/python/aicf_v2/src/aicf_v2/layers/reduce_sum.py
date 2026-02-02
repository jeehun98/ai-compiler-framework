from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec

from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.reduce_sum import reduce_sum as emit_reduce_sum


class ReduceSum(Layer):
    """
    ReduceSum over axis=0 for 2D input (M,N) -> (N,) in f32.

    Kernel contract:
      inputs : [dY]
      outputs: [dB]  (f32)
      schema : 'RSUM'
      payload: int64 axis
    """

    def __init__(self, name: str, *, axis: int = 0):
        super().__init__(name)
        self.axis = int(axis)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        x_spec = b.values[x].spec
        if len(x_spec.shape) != 2:
            raise ValueError(f"ReduceSum expects 2D (M,N); got shape={x_spec.shape}")

        M, N = x_spec.shape  # noqa: F841 (M is unused, kept for clarity)

        # contract: axis must be 0 (sum over M -> N)
        if self.axis != 0:
            raise ValueError(f"ReduceSum only supports axis=0; got axis={self.axis}")

        y = b.value(
            f"{self.name}.out",
            TensorSpec(shape=(N,), dtype="f32", device=x_spec.device),
        )

        # ✅ emitter가 schema/blob/ids까지 채움
        emit_reduce_sum(
            b, ctx,
            x=x,
            out=y,
            axis=self.axis,
            name=f"{self.name}.reduce_sum",
        )
        return y
