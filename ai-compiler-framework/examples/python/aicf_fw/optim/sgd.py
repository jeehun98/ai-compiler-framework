# examples/python/aicf_fw/optim/sgd.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import torch

from .base import Optimizer
from ..modules.base import Parameter


@dataclass
class SGD(Optimizer):
    lr: float = 1e-2

    def step(self, params: Iterable[Parameter]) -> None:
        with torch.no_grad():
            for p in params:
                g = p.grad
                if g is None:
                    continue
                p.data.add_(g, alpha=-self.lr)
