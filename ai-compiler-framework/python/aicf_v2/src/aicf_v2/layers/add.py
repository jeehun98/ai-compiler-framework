from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec

class Add(Layer):
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, a: int, c: int) -> int:
        a_spec = b.values[a].spec
        c_spec = b.values[c].spec
        if a_spec != c_spec:
            raise ValueError(f"Add spec mismatch: {a_spec} vs {c_spec}")

        y = b.value(f"{self.name}.out", a_spec)
        b.emit(
            "add",
            inputs=[a, c],
            outputs=[y],
            name=f"{self.name}.add",
        )
        return y
