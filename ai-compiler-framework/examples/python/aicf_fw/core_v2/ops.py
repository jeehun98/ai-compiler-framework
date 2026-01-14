# aicf_fw/core_v2/ops.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch

from .trace import is_tracing, get_ir, as_ir_value_obj


@dataclass
class SymTensor:
    """
    core_v2용 '심볼릭 텐서'
    - data는 없음 (실제 계산 안 함)
    - meta만 보유
    """
    name: str
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


def sym_tensor(*, name: str, shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> SymTensor:
    return SymTensor(name=str(name), shape=tuple(shape), dtype=dtype, device=device)


def _val(t: SymTensor, fallback: str):
    # SymTensor -> IRValue (obj identity 기반)
    return as_ir_value_obj(t, name=t.name or fallback, shape=t.shape, dtype=t.dtype, device=t.device)


# ============================================================
# Forward ops (Stage1~4)
# ============================================================

def linear(x: SymTensor, W: SymTensor, b: Optional[SymTensor] = None, *, name: str = "linear_out") -> SymTensor:
    """
    y = x @ W^T + b
    shapes:
      x: (B, IN)
      W: (OUT, IN)
      b: (OUT,)
      y: (B, OUT)
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.linear(): tracing-only (no execution in v2 stage-1)")

    if len(x.shape) != 2 or len(W.shape) != 2:
        raise RuntimeError(f"linear expects 2D x/W. got x={x.shape} W={W.shape}")
    B, IN = x.shape
    OUT, IN2 = W.shape
    if IN != IN2:
        raise RuntimeError(f"linear shape mismatch: x={x.shape} W={W.shape} (W must be (OUT, IN))")

    ir = get_ir()

    xv = _val(x, "x")
    wv = _val(W, "W")
    ins = [xv, wv]
    attrs = {"bias": bool(b is not None), "layout": "y = x @ W^T + b"}

    if b is not None:
        bv = _val(b, "b")
        ins.append(bv)

    y = sym_tensor(name=name, shape=(B, OUT), dtype=x.dtype, device=x.device)
    yv = _val(y, name)
    ir.emit(op="Linear", inputs=ins, outputs=[yv], attrs=attrs)
    return y


def relu(x: SymTensor, *, name: str = "relu_out") -> SymTensor:
    if not is_tracing():
        raise RuntimeError("core_v2.ops.relu(): tracing-only")
    ir = get_ir()
    xv = _val(x, "x")
    y = sym_tensor(name=name, shape=x.shape, dtype=x.dtype, device=x.device)
    yv = _val(y, name)
    ir.emit(op="ReLU", inputs=[xv], outputs=[yv], attrs={})
    return y


def mse_grad(
    pred: SymTensor,
    target: SymTensor,
    *,
    scale: Optional[float] = None,
    name: str = "mse_grad_out",
) -> SymTensor:
    """
    out = d/dpred mean((pred-target)^2)
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.mse_grad(): tracing-only")
    if pred.shape != target.shape:
        raise RuntimeError(f"mse_grad shape mismatch: pred={pred.shape} target={target.shape}")

    ir = get_ir()
    pv = _val(pred, "pred")
    tv = _val(target, "target")
    out = sym_tensor(name=name, shape=pred.shape, dtype=pred.dtype, device=pred.device)
    ov = _val(out, name)
    attrs = {}
    if scale is not None:
        attrs["scale"] = float(scale)
    ir.emit(op="MseGrad", inputs=[pv, tv], outputs=[ov], attrs=attrs)
    return out


# ============================================================
# Backward helpers (Stage5A)
# ============================================================

def save(x: SymTensor, *, name: str = "saved") -> SymTensor:
    """
    Save forward activation for backward.
    Lowering: Save -> copy
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.save(): tracing-only")
    ir = get_ir()
    xv = _val(x, "x")
    s = sym_tensor(name=name, shape=x.shape, dtype=x.dtype, device=x.device)
    sv = _val(s, name)
    ir.emit(op="Save", inputs=[xv], outputs=[sv], attrs={})
    return s


def relu_bwd(dout: SymTensor, saved_y: SymTensor, *, name: str = "relu_bwd_out") -> SymTensor:
    """
    d_in = relu_bwd(d_out, saved_y)
    Lowering: ReluBwd -> relu_bwd
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.relu_bwd(): tracing-only")
    if dout.shape != saved_y.shape:
        raise RuntimeError(f"relu_bwd shape mismatch: dout={dout.shape} saved={saved_y.shape}")

    ir = get_ir()
    dv = _val(dout, "dout")
    sv = _val(saved_y, "saved_y")

    din = sym_tensor(name=name, shape=dout.shape, dtype=dout.dtype, device=dout.device)
    iv = _val(din, name)
    ir.emit(op="ReluBwd", inputs=[dv, sv], outputs=[iv], attrs={})
    return din


def linear_bwd(
    x: SymTensor,
    W: SymTensor,
    dY: SymTensor,
    *,
    dx_name: str = "dx",
    dW_name: str = "dW",
    db_name: Optional[str] = "db",
) -> tuple[SymTensor, SymTensor, Optional[SymTensor]]:
    """
    Backward for y = x @ W^T + b

    Inputs:
      x:  (B, IN)
      W:  (OUT, IN)
      dY: (B, OUT)

    Outputs:
      dX: (B, IN)     = dY @ W
      dW: (OUT, IN)   = dY^T @ x
      db: (OUT,)      = reduce_sum(dY, axis=0)    (optional)

    Lowering expectation:
      gemm(dY, W) -> dX                     (transA=False, transB=False)
      gemm(dY, x) -> dW                     (transA=True,  transB=False)  == dY^T @ x
      reduce_sum(dY, axis=0) -> db          (optional)
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.linear_bwd(): tracing-only")

    if len(x.shape) != 2 or len(W.shape) != 2 or len(dY.shape) != 2:
        raise RuntimeError(f"linear_bwd expects 2D tensors. got x={x.shape} W={W.shape} dY={dY.shape}")

    B, IN = x.shape
    OUT, IN2 = W.shape
    B2, OUT2 = dY.shape
    if IN != IN2 or B != B2 or OUT != OUT2:
        raise RuntimeError(f"linear_bwd shape mismatch: x={x.shape} W={W.shape} dY={dY.shape}")

    ir = get_ir()
    xv = _val(x, "x")
    wv = _val(W, "W")
    gv = _val(dY, "dY")

    dx = sym_tensor(name=dx_name, shape=(B, IN), dtype=x.dtype, device=x.device)
    dW = sym_tensor(name=dW_name, shape=(OUT, IN), dtype=W.dtype, device=W.device)

    dxv = _val(dx, dx_name)
    dWv = _val(dW, dW_name)

    outs = [dxv, dWv]
    attrs = {"bias": bool(db_name is not None), "layout": "y = x @ W^T + b"}

    db_t: Optional[SymTensor] = None
    if db_name is not None:
        db_t = sym_tensor(name=db_name, shape=(OUT,), dtype=W.dtype, device=W.device)
        dbv = _val(db_t, db_name)
        outs.append(dbv)

    ir.emit(op="LinearBwd", inputs=[xv, wv, gv], outputs=outs, attrs=attrs)
    return dx, dW, db_t
