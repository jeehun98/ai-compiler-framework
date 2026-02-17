# python/aicf_v2/src/aicf_v2/layers/softmax.py
from __future__ import annotations

from .base import Layer
from ..tensor_spec import TensorSpec
from ..emitters.cuda.context import CudaEmitContext
from ..emitters.cuda.softmax import softmax as emit_softmax

class Softmax(Layer):
    def __init__(self, axis: int = -1, name: str = "softmax"):
        """
        Softmax Layer
        :param axis: Softmax를 적용할 차원 (기본값: 마지막 차원)
        :param name: 레이어 이름
        """
        super().__init__(name)
        self.axis = axis

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        """
        그래프 빌더(b)를 사용하여 Softmax Op를 IR에 등록합니다.
        """
        x_val = b.values[x]
        x_spec = x_val.spec

        # Output Tensor 정의 (Softmax는 입력과 출력의 Shape/Dtype이 동일함)
        y_spec = TensorSpec(
            shape=x_spec.shape, 
            dtype=x_spec.dtype, 
            device=x_spec.device
        )
        y = b.value(f"{self.name}.out", y_spec)

        # Emitter를 호출하여 실제 Op를 그래프에 삽입
        emit_softmax(
            b, ctx,
            x=x,
            out=y,
            axis=self.axis,
            name=self.name,
            # Softmax는 메모리 효율을 위해 Inplace가 가능하도록 제약 조건 부여 가능
            constraints={"inplace_ok": False},
            hints={"prefer_fused_reduction": True} # 커널 최적화 힌트
        )

        return y