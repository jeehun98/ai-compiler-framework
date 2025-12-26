# aicf_fw/nn/losses.py
from __future__ import annotations
from ..core.module import Module
from ..core.tensor import Tensor
from ..core import functional as F

class MSELoss(Module):
    def __init__(self):
        super().__init__()

    def forward(self, y: Tensor, t: Tensor) -> Tensor:
        return F.mse_loss(y, t)
