# aicf_fw/nn/sequential.py
from __future__ import annotations

from aicf_fw.core.module import Module


class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = []
        for i, m in enumerate(layers):
            self.layers.append(m)
            self.add_module(str(i), m)

    def forward(self, x):
        for m in self.layers:
            x = m(x)
        return x
