from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any

import torch

from aicf_fw.core.tensor import Tensor
from aicf_fw.core.autograd import Node, grad_enabled, in_capture
from aicf_fw.backend import get_backend


# ============================================================
# Buffer Pool (capture-safe)
# - CUDA Graph replay requires pointer stability.
# - Therefore: NEVER allocate new tensors in capture region.
# - We cache buffers keyed by (shape, dtype, device, tag).
# ============================================================

class _BufferPool:
    def __init__(self):
        # key: (tag, shape_tuple, dtype, device_index)
        self._bufs: Dict[Tuple[str, Tuple[int, ...], torch.dtype, int], torch.Tensor] = {}

    def _key(self, *, tag: str, ref: torch.Tensor, shape: Tuple[int, ...]) -> Tuple[str, Tuple[int, ...], torch.dtype, int]:
        if not ref.is_cuda:
            raise RuntimeError("BufferPool expects CUDA tensors only.")
        dev = int(ref.device.index) if ref.device.index is not None else 0
        return (tag, tuple(shape), ref.dtype, dev)

    def has(self, *, tag: str, ref: torch.Tensor, shape: Tuple[int, ...]) -> bool:
        key = self._key(tag=tag, ref=ref, shape=shape)
        return key in self._bufs

    def stats(self) -> Dict[str, int]:
        return {"buffers": len(self._bufs)}

    def get(self, *, tag: str, ref: torch.Tensor, shape: Tuple[int, ...]) -> torch.Tensor:
        key = self._key(tag=tag, ref=ref, shape=shape)

        buf = self._bufs.get(key)
        if buf is None:
            # ENFORCEMENT: allocation is forbidden during capture
            if in_capture():
                raise RuntimeError(
                    f"BufferPool: attempted to allocate buffer during capture. key={key}. "
                    "Run warmup forward/backward BEFORE capture to materialize all buffers."
                )
            buf = torch.empty(shape, device=ref.device, dtype=ref.dtype)
            self._bufs[key] = buf
        return buf

    def get_like(self, *, tag: str, ref: torch.Tensor) -> torch.Tensor:
        return self.get(tag=tag, ref=ref, shape=tuple(ref.shape))

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
# Linear
# ============================================================

class _LinearNode(Node):
    """
    Inputs:
      x: (B, IN)
      W: (IN, OUT)
      b: (OUT,) optional

    Saved buffers (stable pointers via pool):
      dx_buf: (B, IN)
      dW_buf: (IN, OUT)
      db_buf: (OUT,) optional
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

    def backward(self, out_grad: Tensor) -> List[Optional[Tensor]]:
        bk = get_backend()
        x = self.inputs[0]
        W = self.inputs[1]

        # dx = dY @ W^T  : (B, OUT) @ (OUT, IN) -> (B, IN)
        bk.op_call_out(
            "gemm",
            [out_grad.data, W.data],
            [self.dx_buf.data],
            {"transB": True},
        )

        # dW = X^T @ dY  : (IN, B) @ (B, OUT) -> (IN, OUT)
        bk.op_call_out(
            "gemm",
            [x.data, out_grad.data],
            [self.dW_buf.data],
            {"transA": True},
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
    y = x @ W + b

    Capture-safe rules:
      - forward output buffer comes from pool (stable pointer)
      - backward buffers come from pool (stable pointer)
      - ONLY op_call_out in capture region
    """
    bk = get_backend()

    if x.data.dim() != 2 or W.data.dim() != 2:
        raise RuntimeError(f"linear expects 2D x and W, got x.dim={x.data.dim()} W.dim={W.data.dim()}")

    B, IN = x.data.shape
    IN2, OUT = W.data.shape
    if IN != IN2:
        raise RuntimeError(f"linear shape mismatch: x={tuple(x.data.shape)} W={tuple(W.data.shape)}")

    # forward y buffer (stable)
    y_buf = _POOL.get(tag="linear_y", ref=x.data, shape=(B, OUT))
    bk.op_call_out("gemm", [x.data, W.data], [y_buf], {})

    if b is not None:
        bk.op_call_out("bias_add", [y_buf, b.data], [y_buf], {})

    y = Tensor(y_buf, requires_grad=grad_enabled())
    if not y.requires_grad:
        return y

    # backward buffers (stable)
    dx_buf_t = _POOL.get(tag="linear_dx", ref=x.data, shape=(B, IN))
    dW_buf_t = _POOL.get(tag="linear_dW", ref=x.data, shape=(IN, OUT))
    db_buf_t = _POOL.get(tag="linear_db", ref=x.data, shape=(OUT,)) if b is not None else None

    dx_buf = Tensor(dx_buf_t, requires_grad=False)
    dW_buf = Tensor(dW_buf_t, requires_grad=False)
    db_buf = Tensor(db_buf_t, requires_grad=False) if db_buf_t is not None else None

    y.creator = _LinearNode(x, W, b, dx_buf, dW_buf, db_buf)
    return y


# ============================================================
# ReLU
# ============================================================

class _ReluNode(Node):
    """
    Save y for relu_bwd input: relu_bwd(dout, y) -> dx
    Buffers are stable via pool.
    """
    def __init__(self, x: Tensor, y: Tensor, dx_buf: Tensor):
        super().__init__([x])
        self.y = y
        self.dx_buf = dx_buf

    def backward(self, out_grad: Tensor) -> List[Optional[Tensor]]:
        bk = get_backend()
        bk.op_call_out(
            "relu_bwd",
            [out_grad.data, self.y.data],
            [self.dx_buf.data],
            {},
        )
        return [self.dx_buf]


def relu(x: Tensor) -> Tensor:
    """
    y = relu(x)
    Capture-safe:
      - forward y buffer from pool
      - backward dx buffer from pool
      - op_call_out only
    """
    bk = get_backend()

    y_buf = _POOL.get_like(tag="relu_y", ref=x.data)
    bk.op_call_out("relu", [x.data], [y_buf], {})

    y = Tensor(y_buf, requires_grad=grad_enabled())
    if not y.requires_grad:
        return y

    dx_buf_t = _POOL.get_like(tag="relu_dx", ref=x.data)
    dx_buf = Tensor(dx_buf_t, requires_grad=False)

    y.creator = _ReluNode(x, y, dx_buf)
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
        raise RuntimeError(f"mse_grad shape mismatch: pred={tuple(pred.data.shape)} target={tuple(target.data.shape)}")

    out_buf = _POOL.get_like(tag="mse_grad_out", ref=pred.data)

    attrs = {}
    if scale is not None:
        attrs["scale"] = float(scale)

    bk.op_call_out("mse_grad", [pred.data, target.data], [out_buf], attrs)
    return Tensor(out_buf, requires_grad=False)
