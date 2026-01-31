from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


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

    def emit(self, b, x: int) -> int:
        x_spec = b.values[x].spec
        if len(x_spec.shape) != 2:
            raise ValueError(f"ReduceSum expects 2D (M,N); got shape={x_spec.shape}")

        M, N = x_spec.shape

        # contract: axis must be 0 (sum over M -> N)
        if self.axis != 0:
            raise ValueError(f"ReduceSum only supports axis=0; got axis={self.axis}")

        y = b.value(f"{self.name}.out", TensorSpec(shape=(N,), dtype="f32", device=x_spec.device))

        b.emit(
            "reduce_sum",
            inputs=[x],
            outputs=[y],
            name=f"{self.name}.reduce_sum",
            attrs={"axis": self.axis},
        )
        return y
