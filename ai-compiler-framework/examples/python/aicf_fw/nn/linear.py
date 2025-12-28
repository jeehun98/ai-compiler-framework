# aicf_fw/nn/linear.py
from __future__ import annotations

import torch

from aicf_fw.core.module import Module
from aicf_fw.core.tensor import Tensor
from aicf_fw.core.functional import linear


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: str = "cuda", dtype: torch.dtype = torch.float32):
        super().__init__()
        W = torch.randn(in_features, out_features, device=device, dtype=dtype) * 0.02
        self.W = Tensor(W, requires_grad=True)

        if bias:
            b = torch.zeros(out_features, device=device, dtype=dtype)
            self.b = Tensor(b, requires_grad=True)
        else:
            self.b = None  # not registered

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.W, self.b)
