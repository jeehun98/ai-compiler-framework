# aicf_fw/core/functional.py
from __future__ import annotations

from typing import List, Optional, Tuple, Dict, Any
import torch

from aicf_fw.core.tensor import Tensor, TensorMeta
from aicf_fw.core.autograd import Node, grad_enabled, in_capture
from aicf_fw.backend import get_backend

from aicf_fw.core.trace import is_tracing, get_ir


# ============================================================
# Buffer Pool (capture-safe)
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
            if in_capture():
                raise RuntimeError(
                    f"BufferPool: attempted to allocate buffer during capture. key={key}. "
                    "Run warmup BEFORE capture."
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


_POOL = _BufferPool()


def reset_functional_buffers() -> None:
    """Drop cached buffers between experiments. Call ONLY outside capture."""
    _POOL.reset()


def functional_buffer_stats() -> Dict[str, int]:
    """Debug: how many buffers currently cached."""
    return _POOL.stats()


# ============================================================
# Tracing helpers
# ============================================================

def _sym_tensor(
    *,
    name: str,
    shape: Tuple[int, ...],
    dtype: torch.dtype,
    device: torch.device,
    requires_grad: bool,
) -> Tensor:
    meta = TensorMeta(shape=tuple(shape), dtype=dtype, device=device)
    return Tensor(None, requires_grad=requires_grad, name=name, meta=meta)


# NOTE: tracing에서 Tensor -> IRValue를 매번 new_value로 만들면 values가 폭증함.
# v0에서는 "Tensor 인스턴스 1개당 IRValue 1개" 정도면 충분.
# cache key는 id(Tensor)로 잡는다.
_TRACE_VAL_CACHE: Dict[int, Any] = {}  # Any = IRValue


def _trace_reset_cache():
    _TRACE_VAL_CACHE.clear()


def _as_ir_value(t: Tensor, fallback_name: str):
    """
    Return cached IRValue for a Tensor while tracing.
    """
    if not is_tracing():
        raise RuntimeError("_as_ir_value() called outside tracing")

    tid = id(t)
    v = _TRACE_VAL_CACHE.get(tid)
    if v is not None:
        return v

    ir = get_ir()
    nm = t.name or fallback_name
    v = ir.new_value(name=nm, shape=t.shape, dtype=str(t.dtype), device=str(t.device))
    _TRACE_VAL_CACHE[tid] = v
    return v


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

    Tracing mode:
      emit high-level IR op "Linear" and return symbolic tensor.
    """
    # ---- TRACING PATH ----
    if is_tracing():
        if len(x.shape) != 2 or len(W.shape) != 2:
            raise RuntimeError(f"linear(trace) expects 2D, got x.shape={x.shape} W.shape={W.shape}")

        B, IN = x.shape
        OUT, IN2 = W.shape
        if IN != IN2:
            raise RuntimeError(f"linear(trace) shape mismatch: x={x.shape} W={W.shape}")

        ir = get_ir()

        xv = _as_ir_value(x, "x")
        wv = _as_ir_value(W, "W")
        inputs = [xv, wv]
        if b is not None:
            bv = _as_ir_value(b, "b")
            inputs.append(bv)

        y = _sym_tensor(
            name="linear_out",
            shape=(B, OUT),
            dtype=x.dtype,
            device=x.device,
            requires_grad=grad_enabled(),
        )
        yv = _as_ir_value(y, "linear_out")

        attrs = {"bias": (b is not None), "layout": "y = x @ W^T + b"}
        ir.emit(op="Linear", inputs=inputs, outputs=[yv], attrs=attrs)
        return y

    # ---- EXECUTION PATH (runtime) ----
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

    Tracing mode:
      emit high-level IR op "ReLU" and return symbolic tensor.
    """
    if is_tracing():
        ir = get_ir()

        xv = _as_ir_value(x, "x")

        y = _sym_tensor(
            name="relu_out",
            shape=x.shape,
            dtype=x.dtype,
            device=x.device,
            requires_grad=grad_enabled(),
        )
        yv = _as_ir_value(y, "relu_out")

        ir.emit(op="ReLU", inputs=[xv], outputs=[yv], attrs={})
        return y

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

    # dedicated saved y buffer keyed by input pointer
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
    default scale = 2/numel

    Tracing mode:
      emit "MseGrad" and return symbolic tensor.
    """
    if is_tracing():
        if pred.shape != target.shape:
            raise RuntimeError(f"mse_grad(trace) shape mismatch: pred={pred.shape} target={target.shape}")

        ir = get_ir()
        pv = _as_ir_value(pred, "pred")
        tv = _as_ir_value(target, "target")

        out = _sym_tensor(
            name="mse_grad_out",
            shape=pred.shape,
            dtype=pred.dtype,
            device=pred.device,
            requires_grad=False,
        )
        ov = _as_ir_value(out, "mse_grad_out")

        attrs = {}
        if scale is not None:
            attrs["scale"] = float(scale)

        ir.emit(op="MseGrad", inputs=[pv, tv], outputs=[ov], attrs=attrs)
        return out

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


# ============================================================
# Optim helpers (capture-safe)
# ============================================================

def grad_zero_(g: Tensor) -> Tensor:
    """
    In-place grad reset (capture-safe).
    Tracing: do nothing (zero_grad is not required in IR v0).
    """
    if is_tracing():
        return g

    bk = get_backend()
    bk.op_call_out("grad_zero", [g.data], [g.data], attrs={})
    return g


def step_inc_(step_i32: torch.Tensor) -> torch.Tensor:
    if is_tracing():
        ir = get_ir()
        sv = ir.new_value(name="step", shape=tuple(step_i32.shape), dtype=str(step_i32.dtype), device=str(step_i32.device))
        ir.emit(op="StepInc", inputs=[sv], outputs=[sv], attrs={})
        return step_i32

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
    if is_tracing():
        ir = get_ir()
        sv = ir.new_value(name="step", shape=tuple(step_i32.shape), dtype=str(step_i32.dtype), device=str(step_i32.device))
        b1 = ir.new_value(name="bc1_inv", shape=tuple(bc1_inv.shape), dtype=str(bc1_inv.dtype), device=str(bc1_inv.device))
        b2 = ir.new_value(name="bc2_inv", shape=tuple(bc2_inv.shape), dtype=str(bc2_inv.dtype), device=str(bc2_inv.device))
        ir.emit(op="BiasCorr", inputs=[sv], outputs=[b1, b2], attrs={"beta1": float(beta1), "beta2": float(beta2)})
        return bc1_inv, bc2_inv

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
    if is_tracing():
        ir = get_ir()
        pv = ir.new_value(name=p.name or "p", shape=p.shape, dtype=str(p.dtype), device=str(p.device))
        gv = ir.new_value(name=g.name or "g", shape=g.shape, dtype=str(g.dtype), device=str(g.device))
        mv = ir.new_value(name=m.name or "m", shape=m.shape, dtype=str(m.dtype), device=str(m.device))
        vv = ir.new_value(name=v.name or "v", shape=v.shape, dtype=str(v.dtype), device=str(v.device))
        b1 = ir.new_value(name="bc1_inv", shape=tuple(bc1_inv.shape), dtype=str(bc1_inv.dtype), device=str(bc1_inv.device))
        b2 = ir.new_value(name="bc2_inv", shape=tuple(bc2_inv.shape), dtype=str(bc2_inv.dtype), device=str(bc2_inv.device))
        ir.emit(
            op="AdamStep",
            inputs=[pv, gv, mv, vv, b1, b2],
            outputs=[pv, mv, vv],
            attrs={"lr": float(lr), "beta1": float(beta1), "beta2": float(beta2), "eps": float(eps)},
        )
        return

    bk = get_backend()
    bk.op_call_out(
        "adam_step",
        [p.data, g.data, m.data, v.data, bc1_inv, bc2_inv],
        [p.data, m.data, v.data],
        attrs={"lr": float(lr), "beta1": float(beta1), "beta2": float(beta2), "eps": float(eps)},
    )
