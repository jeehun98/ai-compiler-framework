from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec

class ReLU(Layer):
    def __init__(self, name: str, save_for_bwd: bool = False):
        super().__init__(name)
        self.save_for_bwd = bool(save_for_bwd)

    def emit(self, b, x: int) -> int:
        x_spec = b.values[x].spec
        y = b.value(f"{self.name}.out", x_spec)

        saved = [y] if self.save_for_bwd else []
        b.emit(
            "relu",
            inputs=[x],
            outputs=[y],
            name=f"{self.name}.relu",
            saved=saved,
        )
        return y
