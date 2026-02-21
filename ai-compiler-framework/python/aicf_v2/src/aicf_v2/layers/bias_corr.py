from __future__ import annotations
from typing import TYPE_CHECKING, Tuple

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import bias_corr  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class BiasCorr(Layer):
    """
    Adam 편향 보정치 계산 레이어.
    (step: i32) -> (bc1_inv: f32, bc2_inv: f32)
    """
    def __init__(self, name: str, *, beta1: float = 0.9, beta2: float = 0.999):
        super().__init__(name)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)

    def emit(self, b, step: int, *, ctx: CudaEmitContext) -> Tuple[int, int]:
        ss = b.values[step].spec
        
        # v2 규격: 0d 텐서 미지원으로 인해 (1,) 형상의 스칼라 텐서 강제
        if tuple(ss.shape) != (1,):
            raise ValueError(f"BiasCorr expects step shape (1,); got {ss.shape}")

        # 출력 Spec 생성 (보통 f32 스칼라)
        o_spec = TensorSpec(shape=(1,), dtype="f32", device=ss.device)
        bc1 = b.value(f"{self.name}.bc1_inv", o_spec)
        bc2 = b.value(f"{self.name}.bc2_inv", o_spec)

        # 통합된 bias_corr.emit 호출
        bias_corr.emit(
            b, ctx,
            step=step,
            out_bc1_inv=bc1,
            out_bc2_inv=bc2,
            beta1=self.beta1,
            beta2=self.beta2,
            name=f"{self.name}.bias_corr",
        )
        
        return bc1, bc2