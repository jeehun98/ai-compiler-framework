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


# -------------------------
# Stage1/5 ops (existing)
# -------------------------

def linear(x: SymTensor, W: SymTensor, b: Optional[SymTensor] = None, *, name: str = "linear_out") -> SymTensor:
    if not is_tracing():
        raise RuntimeError("core_v2.ops.linear(): tracing-only")

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


def save(x: SymTensor, *, name: str = "saved") -> SymTensor:
    """ReLU backward 등에서 쓰는 saved activation."""
    if not is_tracing():
        raise RuntimeError("core_v2.ops.save(): tracing-only")
    ir = get_ir()
    xv = _val(x, "x")
    y = sym_tensor(name=name, shape=x.shape, dtype=x.dtype, device=x.device)
    yv = _val(y, name)
    ir.emit(op="Save", inputs=[xv], outputs=[yv], attrs={})
    return y


def mse_grad(pred: SymTensor, target: SymTensor, *, scale: Optional[float] = None, name: str = "mse_grad_out") -> SymTensor:
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


def linear_bwd(
    x: SymTensor,
    W: SymTensor,
    dY: SymTensor,
    *,
    bias: bool = True,
    dx_name: str = "d_x",
    dW_name: str = "d_W",
    db_name: str = "d_b",
) -> tuple[SymTensor, SymTensor, Optional[SymTensor]]:
    """
    Linear backward (stage5 형태 유지):
      dx = dY @ W
      dW = dY^T @ x
      db = reduce_sum(dY, axis=0)
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.linear_bwd(): tracing-only")

    if len(x.shape) != 2 or len(W.shape) != 2 or len(dY.shape) != 2:
        raise RuntimeError(f"linear_bwd expects 2D x/W/dY. got x={x.shape} W={W.shape} dY={dY.shape}")

    B, IN = x.shape
    OUT, IN2 = W.shape
    if IN != IN2:
        raise RuntimeError(f"linear_bwd shape mismatch: x={x.shape} W={W.shape}")
    if dY.shape != (B, OUT):
        raise RuntimeError(f"linear_bwd shape mismatch: dY={dY.shape} expected {(B, OUT)}")

    ir = get_ir()
    xv = _val(x, "x")
    wv = _val(W, "W")
    gy = _val(dY, "dY")

    dx = sym_tensor(name=dx_name, shape=(B, IN), dtype=x.dtype, device=x.device)
    dW = sym_tensor(name=dW_name, shape=(OUT, IN), dtype=W.dtype, device=W.device)
    dxv = _val(dx, dx_name)
    dWv = _val(dW, dW_name)

    outs = [dxv, dWv]
    out_tensors = [dx, dW]

    if bias:
        db = sym_tensor(name=db_name, shape=(OUT,), dtype=W.dtype, device=W.device)
        dbv = _val(db, db_name)
        outs.append(dbv)
        out_tensors.append(db)

    ir.emit(
        op="LinearBwd",
        inputs=[xv, wv, gy],
        outputs=outs,
        attrs={"bias": bool(bias), "layout": "y = x @ W^T + b"},
    )

    if bias:
        return out_tensors[0], out_tensors[1], out_tensors[2]
    return out_tensors[0], out_tensors[1], None


def relu_bwd(dY: SymTensor, saved: SymTensor, *, name: str = "d_relu_in") -> SymTensor:
    if not is_tracing():
        raise RuntimeError("core_v2.ops.relu_bwd(): tracing-only")
    if dY.shape != saved.shape:
        raise RuntimeError(f"relu_bwd shape mismatch: dY={dY.shape} saved={saved.shape}")

    ir = get_ir()
    gy = _val(dY, "dY")
    sv = _val(saved, "saved")
    out = sym_tensor(name=name, shape=dY.shape, dtype=dY.dtype, device=dY.device)
    ov = _val(out, name)
    ir.emit(op="ReluBwd", inputs=[gy, sv], outputs=[ov], attrs={})
    return out


# -------------------------
# Stage6 optimizer ops (NEW)
# -------------------------

def step_inc(step_i32: SymTensor, *, name: str = "opt.step") -> SymTensor:
    """
    step_out = step_in + 1
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.step_inc(): tracing-only")
    if step_i32.dtype not in (torch.int32, torch.int64):
        raise RuntimeError(f"step_inc expects int32/int64 tensor, got {step_i32.dtype}")

    ir = get_ir()
    si = _val(step_i32, "step")
    out = sym_tensor(name=name, shape=step_i32.shape, dtype=step_i32.dtype, device=step_i32.device)
    so = _val(out, name)
    ir.emit(op="StepInc", inputs=[si], outputs=[so], attrs={})
    return out


def bias_corr(
    step_i32: SymTensor,
    bc1_inv: SymTensor,
    bc2_inv: SymTensor,
    *,
    beta1: float,
    beta2: float,
    out1_name: str = "opt.bc1_inv_out",
    out2_name: str = "opt.bc2_inv_out",
) -> tuple[SymTensor, SymTensor]:
    """
    (bc1_out, bc2_out) = BiasCorr(step)

    NOTE:
    - 실제 연산은 step만 필요(입력 bc1_inv/bc2_inv는 shape/dtype/device meta 용도)
    - Save/IRValue name 충돌 방지를 위해 output name을 기본적으로 분리한다.
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.bias_corr(): tracing-only")
    ir = get_ir()

    sv = _val(step_i32, "step")

    # outputs: 반드시 서로 다른 name으로 생성해서 Save/name collision을 막는다.
    b1o = sym_tensor(name=out1_name, shape=bc1_inv.shape, dtype=bc1_inv.dtype, device=bc1_inv.device)
    b2o = sym_tensor(name=out2_name, shape=bc2_inv.shape, dtype=bc2_inv.dtype, device=bc2_inv.device)
    b1v = _val(b1o, out1_name)
    b2v = _val(b2o, out2_name)

    ir.emit(op="BiasCorr", inputs=[sv], outputs=[b1v, b2v], attrs={"beta1": float(beta1), "beta2": float(beta2)})
    return b1o, b2o


def adam_step(
    p: SymTensor,
    g: SymTensor,
    m: SymTensor,
    v: SymTensor,
    bc1_inv: SymTensor,
    bc2_inv: SymTensor,
    *,
    lr: float,
    beta1: float,
    beta2: float,
    eps: float,
) -> tuple[SymTensor, SymTensor, SymTensor]:
    """
    In-place semantic:
      p, m, v are updated.
    BUT core_v2는 storage 기반이라:
      outputs vids are same storages (executor env[vid]를 그대로 사용)
    따라서 IR에서 outputs는 (p, m, v)로 명시한다.
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.adam_step(): tracing-only")

    ir = get_ir()
    pv = _val(p, "p")
    gv = _val(g, "g")
    mv = _val(m, "m")
    vv = _val(v, "v")
    b1 = _val(bc1_inv, "bc1_inv")
    b2 = _val(bc2_inv, "bc2_inv")

    ir.emit(
        op="AdamStep",
        inputs=[pv, gv, mv, vv, b1, b2],
        outputs=[pv, mv, vv],
        attrs={"lr": float(lr), "beta1": float(beta1), "beta2": float(beta2), "eps": float(eps)},
    )
    return p, m, v

def sgd_step(
    p: SymTensor,
    g: SymTensor,
    *,
    lr: float,
) -> SymTensor:
    """
    SGD update (in-place semantic):
      p = p - lr * g
    core_v2 storage semantics:
      output is p itself.
    """
    if not is_tracing():
        raise RuntimeError("core_v2.ops.sgd_step(): tracing-only")

    ir = get_ir()
    pv = _val(p, "p")
    gv = _val(g, "g")

    ir.emit(
        op="SgdStep",
        inputs=[pv, gv],
        outputs=[pv],
        attrs={"lr": float(lr)},
    )
    return p
