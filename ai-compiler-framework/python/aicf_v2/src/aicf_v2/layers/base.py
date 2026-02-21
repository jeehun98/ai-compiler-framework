from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from ..emitters.cuda.context import CudaEmitContext


class Layer(ABC):
    """
    AICF 레이어 기저 클래스.
    이제 레이어는 역전파 로직(emit_backward)을 직접 들고 있지 않습니다.
    대신 원자적 Emit 노드들을 Builder에 기록하는 역할만 수행합니다.
    """
    def __init__(self, name: str):
        self.name = str(name)

    @abstractmethod
    def emit(self, b: Any, *args: Any, ctx: CudaEmitContext, **kwargs: Any) -> Any:
        """
        레이어의 Forward 연산을 Builder(b)에 기록합니다.
        
        규약:
        - ctx: Model.add() 등에서 제공하는 백엔드 컨텍스트 (키워드 전용).
        - 내부적으로 통합된 emitter 모듈의 .emit()을 호출해야 합니다.
        """
        raise NotImplementedError