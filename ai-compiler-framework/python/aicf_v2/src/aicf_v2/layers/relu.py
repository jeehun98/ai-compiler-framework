from __future__ import annotations

from .base import Layer
from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.relu import relu as emit_relu


class ReLU(Layer):
    def __init__(self, name: str, save_for_bwd: bool = False):
        super().__init__(name)
        self.save_for_bwd = bool(save_for_bwd)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        x_spec = b.values[x].spec
        y = b.value(f"{self.name}.out", x_spec)

        # training semantics: layer가 결정
        saved = [y] if self.save_for_bwd else []

        # ✅ backend-resolved emit은 emitter에 위임
        emit_relu(
            b, ctx,
            x=x,
            out=y,
            name=f"{self.name}.relu",
            saved=saved,
        )
        return y
