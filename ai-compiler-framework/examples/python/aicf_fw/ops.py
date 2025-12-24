# examples/python/aicf_fw/ops.py
from __future__ import annotations
from typing import Optional, Dict, Any

from .backend import get_backend
from .tensor import Tensor
from .modules.base import Parameter
from . import autograd as AG


def relu(x: Tensor) -> Tensor:
    AG.tape().ops.append(AG.ReluOp(kind="relu", x=x.t.contiguous()))
    return get_backend().relu(x)


def gemm(a: Tensor, b: Tensor, attrs: Optional[Dict[str, Any]] = None) -> Tensor:
    return get_backend().gemm(a, b, attrs=attrs)


def linear(x: Tensor, w: Parameter, b: Optional[Parameter] = None) -> Tensor:
    xt = x.t
    wt = w.data.t
    wT = wt.t().contiguous()

    AG.tape().ops.append(
        AG.LinearOp(
            kind="linear",
            x=xt.contiguous(),
            w_param=w,
            b_param=b,
        )
    )

    return get_backend().gemm(Tensor(xt), Tensor(wT), bias=(b.data if b is not None else None))


def mse(y: Tensor, t: Tensor) -> Tensor:
    return get_backend().mse(y, t)
