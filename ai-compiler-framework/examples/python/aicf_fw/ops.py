# examples/python/aicf_fw/ops.py
from __future__ import annotations
from typing import Optional, Any, Dict

import torch

from .backend import get_backend
from .tensor import Tensor


class _NvtxRange:
    def __init__(self, name: str):
        self.name = name

    def __enter__(self):
        # CUDA 없어도 코드가 죽지 않게
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_push(self.name)

    def __exit__(self, exc_type, exc, tb):
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


def gemm(a: Tensor, b: Tensor, bias: Optional[Tensor] = None, act: Optional[str] = None,
         attrs: Optional[Dict[str, Any]] = None) -> Tensor:
    with _NvtxRange("op::gemm"):
        return get_backend().gemm(a, b, bias=bias, act=act, attrs=attrs)

def relu(x: Tensor) -> Tensor:
    with _NvtxRange("op::relu"):
        return get_backend().relu(x)

def mse(y: Tensor, t: Tensor) -> Tensor:
    with _NvtxRange("op::mse"):
        return get_backend().mse(y, t)
