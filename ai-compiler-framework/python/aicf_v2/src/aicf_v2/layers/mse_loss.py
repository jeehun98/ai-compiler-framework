from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import mse_loss  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class MSELoss(Layer):
    """
    Mean Squared Error Loss 레이어.
    이제 역전파 로직을 직접 관리하지 않고 통합 모듈의 emit에 의존합니다.
    """
    def __init__(self, reduction: str = "mean", name: str = "mse"):
        super().__init__(name)
        self.reduction = reduction

    def emit(self, b, y_pred: int, y_true: int, *, ctx: CudaEmitContext) -> int:
        # 결과값은 항상 스칼라(f32)
        out_spec = TensorSpec(shape=(1,), dtype="f32", device=b.device)
        out_vid = b.value(f"{self.name}.out", out_spec)
        
        # 통합된 mse_loss.emit 호출
        mse_loss.emit(
            b, ctx, 
            pred=y_pred, 
            target=y_true, 
            out=out_vid, 
            reduction=self.reduction,
            name=f"{self.name}.fwd"
        )
        return out_vid