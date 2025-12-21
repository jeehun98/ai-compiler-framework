# examples/python/aicf_fw/losses/mse.py
from __future__ import annotations
from .base import Loss
from ..tensor import Tensor
from .. import ops

class MSE(Loss):
    def forward(self, y: Tensor, t: Tensor) -> Tensor:
        return ops.mse(y, t)
