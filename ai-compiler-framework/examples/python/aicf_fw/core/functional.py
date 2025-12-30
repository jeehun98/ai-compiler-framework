# examples/python/aicf_fw/core/functional.py
from __future__ import annotations

from typing import List, Optional, Tuple, Dict

import torch

from aicf_fw.core.tensor import Tensor
from aicf_fw.core.autograd import Node, grad_enabled, in_capture
from aicf_fw.backend import get_backend


# ============================================================
# Buffer Pool (capture-safe)
# - CUDA Graph replay requires pointer stability.
# - Therefore: NEVER allocate new tensors in capture region.
# - We cache buffers keyed by (tag, shape, dtype, device).
# ============================================================

class _BufferPool:
    def __init__(self):
        # key: (tag, shape_tuple, dtype, device_index)
        self._bufs: Dict[Tuple[str, Tuple[int, ...], torch.dtype, int], torch.Tensor] = {}

    def _key(
        self, *, tag: str, ref: torch.Tensor, shape: Tuple[int, ...]
    ) -> Tuple[str, Tuple[int, ...], torch.dtype, int]:
        if not ref.is_cuda:
            raise RuntimeError("BufferPool expects CUDA tensors only.")
        dev = int(ref.device.index) if ref.device.index is not None else 0
        return (tag, tuple(shape), ref.dtype, dev)

    def has(self, *, tag: str, ref: torch.Tensor, shape: Tuple[int, ...]) -> bool:
        key = self._key(tag=tag, ref=ref, shape=shape)
        return key in self._bufs

    def stats(self) -> Dict[str, int]:
        return {"buffers": len(self._bufs)}

    def get(
        self,
        *,
        tag: str,
        ref: torch.Tensor,
        shape: Tuple[int, ...],
        zero_init: bool = False,
    ) -> torch.Tensor:
        key = self._key(tag=tag, ref=ref, shape=shape)
        buf = self._bufs.get(key)
        if buf is None:
            # ENFORCEMENT: allocation is forbidden during capture
            if in_capture():
                raise RuntimeError(
                    f"BufferPool: attempted to allocate buffer during capture. key={key}. "
                    "Run warmup forward/backward BEFORE capture to materialize all buffers."
                )

            buf = (
                torch.zeros(shape, device=ref.device, dtype=ref.dtype)
                if zero_init
                else torch.empty(shape, device=ref.device, dtype=ref.dtype)
            )
            self._bufs[key] = buf
        return buf

    def get_like(self, *, tag: str, ref: torch.Tensor, zero_init: bool = False) -> torch.Tensor:
        return self.get(tag=tag, ref=ref, shape=tuple(ref.shape), zero_init=zero_init)

    def reset(self) -> None:
        if in_capture():
            raise RuntimeError("BufferPool.reset() is forbidden during capture.")
        self._bufs.clear()


# Global pool instance
_POOL = _BufferPool()


def reset_functional_buffers() -> None:
    """
    Drop cached buffers between experiments.
    Call ONLY outside capture.
    """
    _POOL.reset()


def functional_buffer_stats() -> Dict[str, int]:
    """
    Debug: how many buffers currently cached.
    """
    return _POOL.stats()


# ============================================================
# Linear (Torch-compatible)
# ============================================================

class _LinearNode(Node):
    """
    Torch-compatible Linear backward.

    Inputs:
      x: (B, IN)
      W: (OUT, IN)
      b: (OUT,) optional

    Buffers:
      dx: (B, IN)
      dW: (OUT, IN)
      db: (OUT,) optional
    """
    def __init__(
        self,
        x: Tensor,
        W: Tensor,
        b: Optional[Tensor],
        dx_buf: Tensor,
        dW_buf: Tensor,
        db_buf: Optional[Tensor],
    ):
        super().__init__([x, W] + ([b] if b is not None else []))
        self.has_bias = b is not None
        self.dx_buf = dx_buf
        self.dW_buf = dW_buf
        self.db_buf = db_buf

    def backward(self, out_grad: Tensor):
        bk = get_backend()
        x = self.inputs[0]
        W = self.inputs[1]

        # dx = dY @ W : (B, OUT) @ (OUT, IN) -> (B, IN)
        bk.op_call_out(
            "gemm",
            [out_grad.data, W.data],
            [self.dx_buf.data],
            {"transA": False, "transB": False},
        )

        # dW = dY^T @ X : (OUT, B) @ (B, IN) -> (OUT, IN)
        bk.op_call_out(
            "gemm",
            [out_grad.data, x.data],
            [self.dW_buf.data],
            {"transA": True, "transB": False},
        )

        if self.has_bias:
            # db = sum(dY, axis=0) -> (OUT,)
            bk.op_call_out(
                "reduce_sum",
                [out_grad.data],
                [self.db_buf.data],
                {"axis": 0, "keepdim": False},
            )
            return [self.dx_buf, self.dW_buf, self.db_buf]

        return [self.dx_buf, self.dW_buf]


def linear(x: Tensor, W: Tensor, b: Optional[Tensor]) -> Tensor:
    """
    Torch-compatible:
      y = x @ W^T + b

    x: (B, IN)
    W: (OUT, IN)
    b: (OUT,)
    """
    bk = get_backend()

    if x.data.dim() != 2 or W.data.dim() != 2:
        raise RuntimeError(
            f"linear expects 2D x and W, got x.dim={x.data.dim()} W.dim={W.data.dim()}"
        )

    B, IN = x.data.shape
    OUT, IN2 = W.data.shape
    if IN != IN2:
        raise RuntimeError(
            f"linear shape mismatch: x={tuple(x.data.shape)} W={tuple(W.data.shape)} (W must be (OUT, IN))"
        )

    # IMPORTANT: unique tags per-layer to avoid collisions across layers
    wid = int(W.data.data_ptr())
    tag_base = f"linear_{wid}"

    # forward y buffer (stable)
    y_buf = _POOL.get(tag=f"{tag_base}_y", ref=x.data, shape=(B, OUT))

    # y = x @ W^T  => gemm(A=x, B=W, transB=True)
    bk.op_call_out("gemm", [x.data, W.data], [y_buf], {"transB": True})

    if b is not None:
        bk.op_call_out("bias_add", [y_buf, b.data], [y_buf], {})

    y = Tensor(y_buf, requires_grad=grad_enabled())
    if not y.requires_grad:
        return y

    # backward buffers (stable) — ALSO unique per-layer
    # NOTE: zero_init=True is critical if gemm uses beta=1 accumulation semantics
    dx_buf_t = _POOL.get(tag=f"{tag_base}_dx", ref=x.data, shape=(B, IN), zero_init=True)
    dW_buf_t = _POOL.get(tag=f"{tag_base}_dW", ref=x.data, shape=(OUT, IN), zero_init=True)
    db_buf_t = _POOL.get(tag=f"{tag_base}_db", ref=x.data, shape=(OUT,), zero_init=True) if b is not None else None


    dx_buf = Tensor(dx_buf_t, requires_grad=False)
    dW_buf = Tensor(dW_buf_t, requires_grad=False)
    db_buf = Tensor(db_buf_t, requires_grad=False) if db_buf_t is not None else None

    y.creator = _LinearNode(x, W, b, dx_buf, dW_buf, db_buf)
    return y


# ============================================================
# ReLU  (backward-safe saved tensor)
# ============================================================

class _ReluNode(Node):
    """
    Save y for relu_bwd input: relu_bwd(dout, y_saved) -> dx

    Important:
      y_saved MUST NOT be a buffer that can be overwritten by later ops.
      We therefore copy forward y into a dedicated pool buffer keyed by x pointer.
    """
    def __init__(self, x: Tensor, y_saved: Tensor, dx_buf: Tensor):
        super().__init__([x])
        self.y_saved = y_saved
        self.dx_buf = dx_buf

    def backward(self, out_grad: Tensor):
        bk = get_backend()
        bk.op_call_out(
            "relu_bwd",
            [out_grad.data, self.y_saved.data],
            [self.dx_buf.data],
            {},
        )
        return [self.dx_buf]


def relu(x: Tensor) -> Tensor:
    """
    y = relu(x)

    Capture-safe:
      - forward y buffer from pool
      - backward must see the exact forward y (not a reused buffer)
      - so we keep a dedicated saved copy for backward via "copy" op
    """
    bk = get_backend()

    # forward output (can be reused downstream)
    y_buf = _POOL.get_like(tag="relu_y", ref=x.data)
    bk.op_call_out("relu", [x.data], [y_buf], {})

    y = Tensor(y_buf, requires_grad=grad_enabled())
    if not y.requires_grad:
        return y

    # backward dx buffer (shape same as x)
    dx_buf_t = _POOL.get_like(tag="relu_dx", ref=x.data, zero_init=True)
    dx_buf = Tensor(dx_buf_t, requires_grad=False)

    # ---- critical: dedicated saved y buffer keyed by input pointer ----
    xid = int(x.data.data_ptr())
    y_saved_t = _POOL.get_like(tag=f"relu_y_saved_{xid}", ref=x.data)
    bk.op_call_out("copy", [y_buf], [y_saved_t], {})
    y_saved = Tensor(y_saved_t, requires_grad=False)

    y.creator = _ReluNode(x, y_saved, dx_buf)
    return y


# ============================================================
# MSE grad (for training) - out-buffer mode
# ============================================================

def mse_grad(pred: Tensor, target: Tensor, *, scale: Optional[float] = None) -> Tensor:
    """
    dPred = (pred - target) * scale
    default scale = 2/numel  (same as your CUDA kernel)
    Capture-safe:
      - output buffer from pool
      - op_call_out only
    """
    bk = get_backend()

    if pred.data.shape != target.data.shape:
        raise RuntimeError(
            f"mse_grad shape mismatch: pred={tuple(pred.data.shape)} target={tuple(target.data.shape)}"
        )

    out_buf = _POOL.get_like(tag="mse_grad_out", ref=pred.data)

    attrs = {}
    if scale is not None:
        attrs["scale"] = float(scale)

    bk.op_call_out("mse_grad", [pred.data, target.data], [out_buf], attrs)
    return Tensor(out_buf, requires_grad=False)


# ------------------------------------------------------------------
# Existing helpers (capture-safe)
# ------------------------------------------------------------------

def grad_zero_(g: Tensor) -> Tensor:
    """
    In-place grad reset (capture-safe). g must be contiguous CUDA.
    """
    bk = get_backend()
    bk.op_call_out("grad_zero", [g.data], [g.data], attrs={})
    return g


def step_inc_(step_i32: torch.Tensor) -> torch.Tensor:
    """
    step_i32: torch.int32 scalar CUDA tensor (shape=()).
    In-place increment (capture-safe).
    """
    bk = get_backend()
    bk.op_call_out("step_inc", [step_i32], [step_i32], attrs={})
    return step_i32


def bias_corr_out(
    step_i32: torch.Tensor,
    bc1_inv: torch.Tensor,
    bc2_inv: torch.Tensor,
    beta1: float,
    beta2: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute bias-correction inverses into bc tensors (shape=()).
      bc1_inv = 1 / (1 - beta1^t)
      bc2_inv = 1 / (1 - beta2^t)
    step_i32: int32 scalar CUDA tensor, bc*: float32 scalar CUDA tensors.
    """
    bk = get_backend()
    bk.op_call_out(
        "bias_corr",
        [step_i32],
        [bc1_inv, bc2_inv],
        attrs={"beta1": float(beta1), "beta2": float(beta2)},
    )
    return bc1_inv, bc2_inv


def adam_step_(
    p: Tensor,
    g: Tensor,
    m: Tensor,
    v: Tensor,
    bc1_inv: torch.Tensor,
    bc2_inv: torch.Tensor,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> None:
    """
    Adam update in-place for p,m,v.
    v1 bc-tensor AdamStep takes:
      inputs  = [P, G, M, V, bc1_inv, bc2_inv]
      outputs = [P, M, V]
    """
    bk = get_backend()
    bk.op_call_out(
        "adam_step",
        [p.data, g.data, m.data, v.data, bc1_inv, bc2_inv],
        [p.data, m.data, v.data],
        attrs={
            "lr": float(lr),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "eps": float(eps),
        },
    )
