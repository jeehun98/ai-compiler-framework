from __future__ import annotations
from .base import Layer
from ..tensor_spec import TensorSpec


class StepInc(Layer):
    """
    StepInc:
      S_out = S + 1   (int32)

    Note:
      kernel supports in-place.
      Alias/inplace decision is made in plan (constraints inplace_ok).
    """

    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, s: int) -> int:
        ss = b.values[s].spec
        if ss.dtype != "i32":
            raise ValueError(f"StepInc expects i32; got {ss.dtype}")

        so = b.value(f"{self.name}.out", TensorSpec(shape=ss.shape, dtype=ss.dtype, device=ss.device))
        b.emit(
            "step_inc",
            inputs=[s],
            outputs=[so],
            name=f"{self.name}.step_inc",
            constraints={"inplace_ok": True},   # ✅ 이거 추가
        )
        return so
