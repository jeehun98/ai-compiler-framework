# examples/python/aicf_fw/autograd.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple, Union, Literal, Any, Dict

import torch
from aicf_cuda import _C

from .modules.base import Parameter


# -----------------------
# Tape (records forward)
# -----------------------
@dataclass
class LinearOp:
    kind: Literal["linear"]
    x: torch.Tensor              # [B, in] torch.Tensor ONLY (forward input, contiguous)
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


def _require_contig(*xs: torch.Tensor, what: str) -> None:
    for x in xs:
        if not x.is_contiguous():
            raise RuntimeError(f"{what}: requires contiguous (got stride={tuple(x.stride())})")


def _require_cuda(x: torch.Tensor, what: str) -> None:
    if not x.is_cuda:
        raise RuntimeError(f"{what}: CUDA tensor required")


def _same_dtype(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    if a.dtype != b.dtype:
        raise RuntimeError(f"{what}: dtype mismatch {a.dtype} vs {b.dtype}")


def _same_device(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    if a.device != b.device:
        raise RuntimeError(f"{what}: device mismatch {a.device} vs {b.device}")


def _same_shape(a: torch.Tensor, b: torch.Tensor, what: str) -> None:
    if tuple(a.shape) != tuple(b.shape):
        raise RuntimeError(f"{what}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")


# -----------------------
# AICF ops (backward-only)
# -----------------------
def aicf_mse_grad(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    dy = d/dy mean((y - t)^2)
    Kernel contract (current): inputs same dtype/shape/contig, output same dtype as input.
    현실적으로 y는 GEMM(TC) 때문에 fp32일 수 있으므로, t를 y dtype로 캐스팅해서 맞춘다.
    """
    y = _contig(_as_torch(y, "mse_grad:y"))
    t = _contig(_as_torch(t, "mse_grad:t"))

    _require_cuda(y, "mse_grad")
    _require_cuda(t, "mse_grad")
    _same_device(y, t, "mse_grad")

    # IMPORTANT: align dtype to y
    if y.dtype != t.dtype:
        # NOTE: eager 단계에서는 OK. capture-safe로 가려면 prepare_capture_batch에서 미리 맞춰둘 것.
        t = t.to(dtype=y.dtype)

    _same_dtype(y, t, "mse_grad")
    _same_shape(y, t, "mse_grad")
    _require_contig(y, t, what="mse_grad")

    dy = torch.empty_like(y).contiguous()
    _C.op_call(_C.OpKind.MseGrad, [y, t], [dy], {})
    return dy


def aicf_relu_bwd(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    dx = dy * (x > 0)
    NOTE: op_call input order must match your ReluBwd launcher contract.
          여기선 (dy, x) -> dx 로 둠.
          만약 네 커널이 (x, dy) 를 기대하면 아래 inputs 순서만 바꿔.
    """
    dy = _contig(_as_torch(dy, "relu_bwd:dy"))
    x  = _contig(_as_torch(x,  "relu_bwd:x"))

    _require_cuda(dy, "relu_bwd")
    _require_cuda(x,  "relu_bwd")
    _same_dtype(dy, x, "relu_bwd")
    _same_device(dy, x, "relu_bwd")
    _require_contig(dy, x, what="relu_bwd")

    dx = torch.empty_like(dy).contiguous()

    # If your kernel contract is (x, dy) -> dx, change to: [x, dy]
    _C.op_call(_C.OpKind.ReluBwd, [dy, x], [dx], {})
    return dx

def aicf_gemm(
    a: torch.Tensor,
    b: torch.Tensor,
    out_shape: Tuple[int, int],
    attrs: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    a = _contig(_as_torch(a, "gemm:a"))
    b = _contig(_as_torch(b, "gemm:b"))

    _require_cuda(a, "gemm")
    _require_cuda(b, "gemm")
    _same_device(a, b, "gemm")
    _require_contig(a, b, what="gemm")

    # --- 핵심: dtype align ---
    if a.dtype != b.dtype:
        # dY(fp32) 기준으로 W(fp16)을 fp32로 맞춘다
        b = b.to(dtype=a.dtype)

    out = torch.empty(out_shape, device=a.device, dtype=a.dtype).contiguous()
    _C.op_call(_C.OpKind.Gemm, [a, b], [out], attrs or {})
    return out



def aicf_reduce_sum(x: torch.Tensor, axis: int, out_shape) -> torch.Tensor:
    """
    ReduceSum last-dim only (matches your C++ launcher constraints):
      - axis: -1 or last only (we force -1)
      - input:  contiguous, rank>=2
      - output: contiguous, shape == out_shape (caller decides)
    """
    xt = _contig(_as_torch(x, "reduce_sum:x"))
    _require_cuda(xt, "reduce_sum")

    if xt.ndim == 0:
        raise RuntimeError("reduce_sum: scalar not supported")
    if xt.ndim == 1:
        # [N] -> [1, N] (rank>=2 contract)
        xt = xt.view(1, xt.shape[0])
    xt = _contig(xt)

    axis = -1  # force last-dim
    out = torch.empty(out_shape, device=xt.device, dtype=xt.dtype).contiguous()
    _C.op_call(_C.OpKind.ReduceSum, [xt], [out], {"axis": int(axis)})
    return out
