# examples/python/aicf_fw/optim/base.py
from __future__ import annotations
from typing import Iterable
from ..modules.base import Parameter


class Optimizer:
    def step(self, params: Iterable[Parameter]) -> None:
        raise NotImplementedError

    def zero_grad(self, params: Iterable[Parameter]) -> None:
        for p in params:
            p.zero_grad()
