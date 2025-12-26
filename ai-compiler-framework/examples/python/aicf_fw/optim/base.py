from __future__ import annotations
from typing import Iterable
import torch

from ..modules.base import Parameter


class Optimizer:
    def zero_grad(self, params: Iterable[Parameter]) -> None:
        # Parameter 구현이 여러 군데여도 동작하도록 "data.t.grad" 직접 처리
        with torch.no_grad():
            for p in params:
                if not hasattr(p, "data") or not hasattr(p.data, "t"):
                    continue
                if p.data.t.grad is not None:
                    p.data.t.grad = None

    def step(self, params: Iterable[Parameter]) -> None:
        raise NotImplementedError
