from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import softmax  # 통합된 softmax 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class Softmax(Layer):
    """
    Softmax 활성화 함수 레이어.
    이제 역전파 로직을 직접 관리하지 않고 통합 모듈의 emit에 의존합니다.
    """
    def __init__(self, axis: int = -1, name: str = "softmax"):
        super().__init__(name)
        self.axis = axis

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        """Forward: y = softmax(x)"""
        x_spec = b.values[x].spec
        y_spec = TensorSpec(shape=x_spec.shape, dtype=x_spec.dtype, device=x_spec.device)
        y = b.value(f"{self.name}.out", y_spec)

        # 통합된 softmax.emit 호출
        softmax.emit(
            b, ctx,
            x=x,
            out=y,
            axis=self.axis,
            name=self.name
        )
        return y