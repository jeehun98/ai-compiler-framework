# examples/python/aicf_fw/modules/activations.py
from __future__ import annotations
from .base import Module
from ..tensor import Tensor
from .. import ops

class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)
