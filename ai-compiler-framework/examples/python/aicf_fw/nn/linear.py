# aicf_fw/nn/linear.py
from __future__ import annotations
import torch
from ..core.module import Module
from ..core.tensor import Parameter, Tensor
from ..core import functional as F

class Linear(Module):
    def __init__(self, in_dim: int, out_dim: int, bias: bool = True, dtype=None, device=None):
        super().__init__()
        w = torch.randn(in_dim, out_dim, dtype=dtype, device=device) * 0.02
        self.W = Parameter(w, name="W")
        if bias:
            b = torch.zeros(out_dim, dtype=dtype, device=device)
            self.b = Parameter(b, name="b")
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.W, self.b)
