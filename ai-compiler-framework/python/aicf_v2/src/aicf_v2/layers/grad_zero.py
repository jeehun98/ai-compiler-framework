from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


class GradZero(Layer):
    """
    GradZero:
      y = 0 (same shape/dtype/device)

    Kernel contract:
      inputs : [x]
      outputs: [y]
      schema : 0
      payload: empty

    Note:
      kernel supports in-place (y aliases x), but v2 builder currently tracks
      producer_op per Value; true in-place would require 'last_writer' semantics.
      So we keep it out-of-place for now.
    """

    def __init__(self, name: str, *, inplace: bool = False):
        super().__init__(name)
        self.inplace = bool(inplace)

    def emit(self, b, x: int) -> int:
        xs = b.values[x].spec

        if self.inplace:
            raise NotImplementedError(
                "GradZero(inplace=True) requires last-writer tracking (producer_op overwrite issue). "
                "Use inplace=False for now."
            )

        y = b.value(f"{self.name}.out", TensorSpec(shape=xs.shape, dtype=xs.dtype, device=xs.device))
        b.emit(
            "grad_zero",
            inputs=[x],
            outputs=[y],
            name=f"{self.name}.grad_zero",
        )
        return y
