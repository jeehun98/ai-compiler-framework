from __future__ import annotations
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..model import Model

class Optimizer(ABC):
    def __init__(self, model: Model):
        self.model = model

    @abstractmethod
    def step(self):
        """그래프에 업데이트 연산들을 추가합니다."""
        pass