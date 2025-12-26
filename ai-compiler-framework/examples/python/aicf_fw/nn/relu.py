# aicf_fw/nn/relu.py
from __future__ import annotations
from ..core.module import Module
from ..core.tensor import Tensor
from ..core import functional as F

class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        return F.relu(x)
