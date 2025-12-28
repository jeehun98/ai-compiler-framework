# aicf_fw/nn/relu.py
from __future__ import annotations

from aicf_fw.core.module import Module
from aicf_fw.core.tensor import Tensor
from aicf_fw.core.functional import relu


class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return relu(x)
