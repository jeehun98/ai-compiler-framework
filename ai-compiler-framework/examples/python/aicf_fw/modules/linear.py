from __future__ import annotations
from typing import Optional
import torch

from .base import Module
from ..tensor import Tensor, Parameter
from .. import ops


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        w = torch.empty((out_features, in_features), device="cuda", dtype=torch.float32).contiguous()
        torch.nn.init.kaiming_uniform_(w, a=5**0.5)
        self.weight = Parameter(Tensor(w))

        if bias:
            b = torch.zeros((out_features,), device="cuda", dtype=torch.float32).contiguous()
            self.bias = Parameter(Tensor(b))
        else:
            self.bias = None

        # register (Sequential.parameters traversal 위해)
        self.add_parameter("weight", self.weight)
        if self.bias is not None:
            self.add_parameter("bias", self.bias)

    def forward(self, x: Tensor) -> Tensor:
        return ops.linear(x, self.weight, self.bias if self.bias is not None else None)
