# examples/python/aicf_fw/ops.py
from __future__ import annotations
from typing import Optional, Dict, Any

from .backend import get_backend
from .tensor import Tensor
from .modules.base import Parameter
from . import autograd as AG


def relu(x: Tensor) -> Tensor:
    # tape에는 contig만 기록 (eager에서만 쓰더라도 일관성 유지)
    AG.tape().ops.append(AG.ReluOp(kind="relu", x=x.t.contiguous()))
    return get_backend().relu(x)


def gemm(a: Tensor, b: Tensor, attrs: Optional[Dict[str, Any]] = None) -> Tensor:
    return get_backend().gemm(a, b, attrs=attrs)


def linear(x: Tensor, w: Parameter, b: Optional[Parameter] = None) -> Tensor:
    xC = x.contiguous()
    wC = w.data.contiguous()

    AG.tape().ops.append(
        AG.LinearOp(
            kind="linear",
            x=xC,
            w_param=w,
            b_param=b,
        )
    )

    # IMPORTANT:
    # Our GEMM backend expects:
    #   y = x @ (w_storage)^T  with attrs transB=True
    # where w_storage is [Dout, Din] (row-major contiguous).
    attrs = {"transA": False, "transB": True}

    return get_backend().gemm(
        xC,
        wC,
        bias=(b.data if b is not None else None),
        attrs=attrs,
    )


def mse(y: Tensor, t: Tensor) -> Tensor:
    return get_backend().mse(y, t)
