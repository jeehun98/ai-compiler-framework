from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import sgd_step  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class SgdStep(Layer):
    """
    SGD 가중치 업데이트 레이어.
    Contract: P_new = P - lr * G
    """
    def __init__(self, name: str, *, lr: float = 1e-3):
        super().__init__(name)
        self.lr = float(lr)

    def emit(self, b, P: int, G: int, *, ctx: CudaEmitContext) -> int:
        P_spec = b.values[P].spec
        
        # 출력 Vid 생성 (Lattice: P와 동일한 spec)
        Pout = b.value(f"{self.name}.P", TensorSpec(shape=P_spec.shape, dtype=P_spec.dtype, device=P_spec.device))

        # 통합된 sgd_step.emit 호출
        sgd_step.emit(
            b, ctx,
            P=P,
            G=G,
            outP=Pout,
            lr=self.lr,
            name=f"{self.name}.sgd_step",
            constraints={"inplace_ok": True},
        )
        return Pout