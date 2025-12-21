from __future__ import annotations
from typing import Optional, Dict, Any
import torch

from .backend import get_backend
from .tensor import Tensor

def relu(x: Tensor) -> Tensor:
    return get_backend().relu(x)

def gemm(a: Tensor, b: Tensor, attrs: Optional[Dict[str, Any]] = None) -> Tensor:
    return get_backend().gemm(a, b, attrs=attrs)

def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    # x: [B, in], w: [out, in] -> need wT: [in, out]
    xt = x.t if hasattr(x, "t") else x
    wt = w.t if hasattr(w, "t") else w
    wT = wt.t().contiguous()  # v0.2가 contig desc만 받으니 여기서 강제
    y = gemm(Tensor(xt), Tensor(wT))
    if b is not None:
        # bias broadcast add는 아직 AICF add가 1:1만 지원이면 torch fallback 가능
        # (나중에 bias add variant 만들면 여기서 op_call로 내림)
        yt = y.t
        bt = b.t if hasattr(b, "t") else b
        y = Tensor(yt + bt)
    return y

def mse(y: Tensor, t: Tensor) -> Tensor:
    return get_backend().mse(y, t)
from __future__ import annotations
from typing import Optional, Dict, Any
import torch

from .backend import get_backend
from .tensor import Tensor

def relu(x: Tensor) -> Tensor:
    return get_backend().relu(x)

def gemm(a: Tensor, b: Tensor, attrs: Optional[Dict[str, Any]] = None) -> Tensor:
    return get_backend().gemm(a, b, attrs=attrs)

def linear(x: Tensor, w: Tensor, b: Optional[Tensor] = None) -> Tensor:
    # x: [B, in], w: [out, in] -> need wT: [in, out]
    xt = x.t if hasattr(x, "t") else x
    wt = w.t if hasattr(w, "t") else w
    wT = wt.t().contiguous()  # v0.2가 contig desc만 받으니 여기서 강제
    y = gemm(Tensor(xt), Tensor(wT))
    if b is not None:
        # bias broadcast add는 아직 AICF add가 1:1만 지원이면 torch fallback 가능
        # (나중에 bias add variant 만들면 여기서 op_call로 내림)
        yt = y.t
        bt = b.t if hasattr(b, "t") else b
        y = Tensor(yt + bt)
    return y

def mse(y: Tensor, t: Tensor) -> Tensor:
    return get_backend().mse(y, t)
