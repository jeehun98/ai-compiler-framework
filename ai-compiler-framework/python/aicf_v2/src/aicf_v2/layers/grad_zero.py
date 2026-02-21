from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import grad_zero  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class GradZero(Layer):
    """
    텐서 버퍼를 0으로 밀어버리는 레이어입니다.
    주로 optimizer.zero_grad() 단계를 IR로 표현할 때 사용합니다.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        xs = b.values[x].spec

        # 출력 Vid 생성 (Lattice: 입력과 동일한 스펙)
        y = b.value(f"{self.name}.out", TensorSpec(shape=xs.shape, dtype=xs.dtype, device=xs.device))

        # 통합된 grad_zero.emit 호출
        grad_zero.emit(
            b, ctx,
            x=x,
            out=y,
            name=f"{self.name}.grad_zero",
            constraints={"inplace_ok": True},
        )
        return y