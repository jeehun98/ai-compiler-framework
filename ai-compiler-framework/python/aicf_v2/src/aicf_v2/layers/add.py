from __future__ import annotations
from typing import TYPE_CHECKING

from .base import Layer
from ..emitters.cuda import add  # 통합 모듈 임포트

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext

class Add(Layer):
    def __init__(self, name: str):
        super().__init__(name)

    def emit(self, b, a: int, c: int, *, ctx: CudaEmitContext) -> int:
        """
        Add 레이어 Forward. 
        규격화된 add.emit을 호출하여 Builder에 노드를 기록합니다.
        """
        a_spec = b.values[a].spec
        c_spec = b.values[c].spec
        
        # 기본적 Spec 검증 (Broadcasting 미지원 시)
        if a_spec.shape != c_spec.shape:
             raise ValueError(f"Add spec mismatch: {a_spec.shape} vs {c_spec.shape}")

        y = b.value(f"{self.name}.out", a_spec)

        # 통합된 add.emit 호출
        add.emit(
            b, ctx,
            a=a,
            c=c,
            out=y,
            name=f"{self.name}.add",
        )
        return y