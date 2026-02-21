from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..emitters.cuda import relu  # 통합된 relu 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class ReLU(Layer):
    """
    ReLU 활성화 함수 레이어.
    이제 역전파 로직(emit_backward)을 직접 들고 있지 않으며,
    통합된 relu.emit 규격에 따라 FWD 노드를 생성하는 역할만 수행합니다.
    """
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, x: int, *, ctx: CudaEmitContext) -> int:
        """Forward: y = relu(x)"""
        x_spec = b.values[x].spec
        
        # 출력 Vid 생성 (Lattice: x와 동일한 spec)
        y = b.value(f"{self.name}.out", x_spec)

        # 통합된 relu.emit 호출
        # 이 호출은 Builder에 'kind="relu"' 노드를 남기며, 
        # 나중에 Mirroring BWD의 이정표가 됩니다.
        relu.emit(
            b, ctx,
            x=x,
            out=y,
            name=f"{self.name}.relu",
        )
        
        return y