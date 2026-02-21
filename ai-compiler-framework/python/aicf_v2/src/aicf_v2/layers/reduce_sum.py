from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda import reduce_sum  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class ReduceSum(Layer):
    """
    ReduceSum 레이어.
    2D 입력 (M, N)에 대해 axis=0 방향으로 합산하여 (N,) 결과를 생성합니다.
    """
    def __init__(self, name: str, *, axis: int = 0):
        super().__init__(name)
        self.axis = int(axis)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        x_spec = b.values[x].spec
        if len(x_spec.shape) != 2:
            raise ValueError(f"ReduceSum expects 2D (M, N); got {x_spec.shape}")

        M, N = x_spec.shape
        if self.axis != 0:
             # 현재 백엔드 제약 사항 반영
             raise ValueError(f"ReduceSum only supports axis=0 in this version; got {self.axis}")

        # 출력 Spec 정의: (M, N) --axis 0--> (N,)
        y_spec = TensorSpec(shape=(N,), dtype="f32", device=x_spec.device)
        y = b.value(f"{self.name}.out", y_spec)

        # 통합된 reduce_sum.emit 호출
        reduce_sum.emit(
            b, ctx,
            x=x,
            out=y,
            axis=self.axis,
            name=f"{self.name}.reduce_sum",
        )
        return y