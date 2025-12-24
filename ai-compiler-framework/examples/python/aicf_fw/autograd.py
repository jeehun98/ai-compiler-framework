# examples/python/aicf_fw/autograd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Literal, Any

import torch
from aicf_cuda import _C

from .modules.base import Parameter


# -----------------------
# Tape (records forward)
# -----------------------
@dataclass
class LinearOp:
    kind: Literal["linear"]
    x: torch.Tensor              # [B, in]  (torch.Tensor ONLY)
    w_param: Parameter           # Parameter(Tensor(torch.Tensor))
    b_param: Optional[Parameter] # Parameter(Tensor(torch.Tensor)) or None


@dataclass
class ReluOp:
    kind: Literal["relu"]
    x: torch.Tensor              # torch.Tensor ONLY (same layout as forward input)


Op = Union[LinearOp, ReluOp]


class Tape:
    def __init__(self) -> None:
        self.ops: List[Op] = []

    def clear(self) -> None:
        self.ops.clear()


_TAPE = Tape()


def tape() -> Tape:
    return _TAPE


# -----------------------
# Helpers
# -----------------------
def _as_torch(x: Any, what: str) -> torch.Tensor:
    # Accept torch.Tensor or wrapper with .t (your Tensor class)
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "t") and isinstance(getattr(x, "t"), torch.Tensor):
        return getattr(x, "t")
    raise TypeError(f"{what}: expected torch.Tensor or Tensor wrapper with .t, got {type(x)}")


def _contig(x: torch.Tensor) -> torch.Tensor:
    return x if x.is_contiguous() else x.contiguous()


def _require_cuda(x: torch.Tensor, what: str) -> None:
    if not x.is_cuda:
        raise RuntimeError(f"{what}: CUDA tensor required")


def _same_dtype(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    if a.dtype != b.dtype:
        raise RuntimeError(f"{what}: dtype mismatch {a.dtype} vs {b.dtype}")


def _same_device(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    if a.device != b.device:
        raise RuntimeError(f"{what}: device mismatch {a.device} vs {b.device}")


# -----------------------
# AICF ops (backward-only)
# -----------------------
def aicf_mse_grad(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    mean MSE: mean((y-t)^2)
    d/dy = 2*(y-t)/numel
    """
    y = _contig(_as_torch(y, "mse_grad:y"))
    t = _contig(_as_torch(t, "mse_grad:t"))
    _require_cuda(y, "mse_grad")
    _require_cuda(t, "mse_grad")
    _same_dtype(y, t, "mse_grad")
    _same_device(y, t, "mse_grad")

    dy = torch.empty_like(y).contiguous()

    scale = 2.0 / float(y.numel())
    _C.op_call(_C.OpKind.MseGrad, [y, t], [dy], {"scale": float(scale)})
    return dy


def aicf_relu_bwd(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    dx = dy * (x > 0)
    NOTE: your ReluBwd test uses inputs [Y, dOut] or [dy, x] depending on kernel contract.
          Here we match your trainer usage: (dy, x) -> dx.
          If your kernel expects (y, dy), swap order in the call below.
    """
    dy = _contig(_as_torch(dy, "relu_bwd:dy"))
    x  = _contig(_as_torch(x,  "relu_bwd:x"))
    _require_cuda(dy, "relu_bwd")
    _require_cuda(x,  "relu_bwd")
    _same_dtype(dy, x, "relu_bwd")
    _same_device(dy, x, "relu_bwd")

    dx = torch.empty_like(dy).contiguous()

    # If your kernel contract is (Y, dOut) -> dY, change to: [x, dy]
    _C.op_call(_C.OpKind.ReluBwd, [dy, x], [dx], {})
    return dx


def aicf_gemm(a: torch.Tensor, b: torch.Tensor, out_shape: Tuple[int, int]) -> torch.Tensor:
    """
    Out = Gemm(a, b)
    Assumes your Gemm op matches the shapes the trainer passes.
    """
    a = _contig(_as_torch(a, "gemm:a"))
    b = _contig(_as_torch(b, "gemm:b"))
    _require_cuda(a, "gemm")
    _require_cuda(b, "gemm")
    _same_dtype(a, b, "gemm")
    _same_device(a, b, "gemm")

    out = torch.empty(out_shape, device=a.device, dtype=a.dtype).contiguous()
    _C.op_call(_C.OpKind.Gemm, [a, b], [out], {})
    return out


def aicf_reduce_sum(x: torch.Tensor, axis: int, out_shape) -> torch.Tensor:
    """
    ReduceSum last-dim only (matches your C++ launcher constraints):
      - input:  f32 contig, rank>=2
      - output: f32 contig, rank==1, shape [N] where N = input.shape[last]
      - axis:   -1 or last (we FORCE -1)
    """
    xt = _as_torch(x, "reduce_sum:x")
    xt = _contig(xt)

    _require_cuda(xt, "reduce_sum")

    # ---- normalize to rank>=2 and last-dim reduction ----
    if xt.ndim == 0:
        raise RuntimeError("reduce_sum: scalar not supported")
    if xt.ndim == 1:
        # [N] -> [1, N]
        xt = xt.view(1, xt.shape[0])
    # if >=2, keep as-is

    xt = _contig(xt)

    # force last-dim axis (kernel supports only last)
    axis = -1

    out = torch.empty(out_shape, device=xt.device, dtype=xt.dtype).contiguous()
    _C.op_call(_C.OpKind.ReduceSum, [xt], [out], {"axis": int(axis)})
    return out
