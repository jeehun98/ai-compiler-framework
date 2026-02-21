from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

from .base import Layer
from ..emitters.cuda import adam_step  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class AdamStep(Layer):
    def __init__(
        self,
        name: str,
        *,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(name)
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

    def emit(self, b, P: int, G: int, M: int, V: int, bc1: int, bc2: int, *, ctx: CudaEmitContext) -> Tuple[int, int, int]:
        """Adam 업데이트 그래프를 빌드합니다."""
        P_spec = b.values[P].spec
        M_spec = b.values[M].spec
        V_spec = b.values[V].spec

        # 1. 출력 Vid 생성 (Lattice: spec을 그대로 복사하여 정밀도 유지)
        Pout = b.value(f"{self.name}.P", P_spec)
        Mout = b.value(f"{self.name}.M", M_spec)
        Vout = b.value(f"{self.name}.V", V_spec)

        # 2. 통합된 adam_step.emit 호출
        adam_step.emit(
            b, ctx,
            P=P, G=G, M=M, V=V,
            bc1=bc1, bc2=bc2,
            outP=Pout, outM=Mout, outV=Vout,
            lr=self.lr,
            beta1=self.beta1,
            beta2=self.beta2,
            eps=self.eps,
            name=f"{self.name}.adam_step"
        )

        return Pout, Mout, Vout