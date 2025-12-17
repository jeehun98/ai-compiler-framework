# examples/python/aicf_fw/modules/linear.py
from __future__ import annotations
from .base import Module, Parameter
from ..tensor import Tensor
from .. import ops
import math
import torch

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        w = Tensor.randn((in_features, out_features), requires_grad=False)
        w = Tensor(w.t * (1.0 / math.sqrt(in_features)))   # 새 텐서 생성
        w.t.requires_grad_(True)                           # leaf로 만들기(권장: detach 사용)

        self.W = Parameter(Tensor(w.t.detach().requires_grad_(True)))
        
        with torch.no_grad():
            self.W.data.t.mul_(1.0 / math.sqrt(in_features))
        self.add_parameter("W", self.W)

        self.b = None
        if bias:
            self.b = Parameter(Tensor.zeros((out_features,), requires_grad=True))
            self.add_parameter("b", self.b)

    def forward(self, x: Tensor) -> Tensor:
        return ops.gemm(x, self.W.data, bias=self.b.data if self.b else None)
