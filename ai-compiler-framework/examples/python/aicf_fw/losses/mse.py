# examples/python/aicf_fw/losses/mse.py
from __future__ import annotations
from .base import Loss
from ..tensor import Tensor
from .. import ops

class MSELoss(Loss):
    def __call__(self, y: Tensor, t: Tensor) -> Tensor:
        return ops.mse(y, t)
