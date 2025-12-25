from __future__ import annotations
from typing import Optional
import os
import torch

from .base import Module
from ..tensor import Tensor, Parameter
from .. import ops


def _env_dtype() -> torch.dtype:
    s = os.environ.get("AICF_DTYPE", "f32").lower()
    return torch.float16 if s == "f16" else torch.float32


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()

        dt = _env_dtype()

        w = torch.empty((out_features, in_features), device="cuda", dtype=dt).contiguous()
        torch.nn.init.kaiming_uniform_(w, a=5**0.5)
        self.weight = Parameter(Tensor(w))

        if bias:
            b = torch.zeros((out_features,), device="cuda", dtype=dt).contiguous()
            self.bias = Parameter(Tensor(b))
        else:
            self.bias = None

        self.add_parameter("weight", self.weight)
        if self.bias is not None:
            self.add_parameter("bias", self.bias)

    # ✅ 이게 반드시 "class Linear" 내부에 있어야 함 (들여쓰기 중요)
    def forward(self, x: Tensor) -> Tensor:
        return ops.linear(x, self.weight, self.bias if self.bias is not None else None)
