# examples/python/aicf_fw/nn/linear.py
from __future__ import annotations
import torch

from ..core.module import Module
from ..core.tensor import Tensor
from ..core import functional as F

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 device: str = "cuda", dtype=torch.float32):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # ✅ 파라미터를 처음부터 CUDA에 생성
        W = torch.randn(in_features, out_features, device=device, dtype=dtype) * 0.02
        self.W = Tensor(W, requires_grad=True)

        if bias:
            b = torch.zeros(out_features, device=device, dtype=dtype)
            self.b = Tensor(b, requires_grad=True)
        else:
            self.b = None

    def forward(self, x: Tensor) -> Tensor:
        return F.linear(x, self.W, self.b)
