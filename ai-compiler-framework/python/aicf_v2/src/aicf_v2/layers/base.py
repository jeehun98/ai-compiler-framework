from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ..emitters.cuda.context import CudaEmitContext


class Layer(ABC):
    def __init__(self, name: str):
        self.name = str(name)

    @abstractmethod
    def emit(self, b, *args: Any, ctx: CudaEmitContext, **kwargs: Any):
        """
        Contract:
          - ctx is keyword-only and provided by Model.add() / Model.apply().
          - emitters are responsible for filling op.kind_id / op.attr_schema / op.attr_blob via emit_resolved.
        """
        raise NotImplementedError
