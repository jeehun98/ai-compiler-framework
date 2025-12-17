# examples/python/aicf_fw/modules/sequential.py
from __future__ import annotations
from typing import Iterable, List

from .base import Module


class Sequential(Module):
    def __init__(self, *layers: Module) -> None:
        super().__init__()
        self.layers: List[Module] = list(layers)
        for i, m in enumerate(self.layers):
            self.add_module(str(i), m)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x
