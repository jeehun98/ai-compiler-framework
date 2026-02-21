from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import mse_grad  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class MseGrad(Layer):
    """
    MSE Gradient 레이어.
    Loss 함수를 거치지 않고 직접 Gradient를 생성할 때 사용합니다.
    """

    def __init__(self, name: str, *, scale: Optional[float] = None):
        super().__init__(name)
        self.scale = None if scale is None else float(scale)

    def emit(self, b, pred: int, target: int, *, ctx: CudaEmitContext) -> int:
        p_spec = b.values[pred].spec
        t_spec = b.values[target].spec

        # 기본적인 Spec 검증
        if p_spec.shape != t_spec.shape:
            raise ValueError(f"MseGrad shape mismatch: {p_spec.shape} vs {t_spec.shape}")

        # 출력 Vid 생성 (pred와 동일한 형상)
        g = b.value(f"{self.name}.g", TensorSpec(shape=p_spec.shape, dtype=p_spec.dtype, device=p_spec.device))

        # 통합된 mse_grad.emit 호출
        mse_grad.emit(
            b, ctx,
            pred=pred,
            target=target,
            out=g,
            scale=self.scale,
            name=f"{self.name}.mse_grad",
        )
        
        return g