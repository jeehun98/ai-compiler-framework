# aicf_fw/nn/sequential.py
from __future__ import annotations
from typing import List
from ..core.module import Module
from ..core.tensor import Tensor

class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers: List[Module] = []
        for i, l in enumerate(layers):
            self.layers.append(l)
            self.add_module(str(i), l)

    def forward(self, x: Tensor) -> Tensor:
        for l in self.layers:
            x = l(x)
        return x
