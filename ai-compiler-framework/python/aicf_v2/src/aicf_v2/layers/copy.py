from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import copy  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class Copy(Layer):
    """
    명시적 텐서 복사 레이어.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        x_spec = b.values[x].spec
        
        # 출력 Vid 생성 (Lattice: x와 동일한 spec)
        y = b.value(
            f"{self.name}.out", 
            TensorSpec(shape=x_spec.shape, dtype=x_spec.dtype, device=x_spec.device)
        )

        # 통합된 copy.emit 호출
        copy.emit(
            b, ctx,
            x=x,
            out=y,
            name=f"{self.name}.copy",
        )
        return y