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


def mse_grad(pred: SymTensor, target: SymTensor, *, scale: Optional[float] = None, name: str = "mse_grad_out") -> SymTensor:
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
