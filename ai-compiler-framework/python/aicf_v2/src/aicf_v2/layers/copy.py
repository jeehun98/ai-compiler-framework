from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


class Copy(Layer):
    """
    Copy:
      y = x  (device-to-device copy, same dtype/shape)

    Kernel contract:
      inputs : [x]
      outputs: [y]
      schema : 0
      payload: empty
    """

    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, x: int) -> int:
        xs = b.values[x].spec
        y = b.value(f"{self.name}.out", TensorSpec(shape=xs.shape, dtype=xs.dtype, device=xs.device))

        b.emit(
            "copy",
            inputs=[x],
            outputs=[y],
            name=f"{self.name}.copy",
        )
        return y
