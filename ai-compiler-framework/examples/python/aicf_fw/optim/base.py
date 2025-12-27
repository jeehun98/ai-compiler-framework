# aicf_fw/optim/base.py
from __future__ import annotations
from typing import List
from ..core.tensor import Tensor

class Optimizer:
    def __init__(self, params: List[Tensor]):
        self.params = params

    def zero_grad(self) -> None:
        for p in self.params:
            if p.grad is not None:
                p.grad = None

    def step(self) -> None:
        raise NotImplementedError
