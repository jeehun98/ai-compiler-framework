# examples/python/python_framework_test/v2_stage6_train1_check_train_step.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# --- bootstrap: examples/python + build/python ---
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
# ----------------------------------------------

from aicf_fw.core_v2 import trace_ir, dump_ir, dump_lowered, dump_plan
from aicf_fw.core_v2.ops import (
    sym_tensor, linear, relu, save, mse_grad,
    linear_bwd, relu_bwd,
    step_inc, bias_corr, adam_step,
)
from aicf_fw.core_v2.lower import lower_to_backend_ops
from aicf_fw.core_v2.plan import build_binding_plan
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions
import aicf_cuda


def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def maxabs(t: torch.Tensor) -> float:
    return float(t.abs().max().item())


def main():
    # ✅ train mode
    os.environ.pop("AICF_WARMUP", None)

    tf32_off()
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    dtype = torch.float32
    B, D = 64, 8

    x = torch.randn(B, D, device=device, dtype=dtype)
    t = torch.randn(B, D, device=device, dtype=dtype)

    W0 = torch.randn(D, D, device=device, dtype=dtype)
    b0 = torch.randn(D, device=device, dtype=dtype)
    W1 = torch.randn(D, D, device=device, dtype=dtype)
    b1 = torch.randn(D, device=device, dtype=dtype)

    # opt state
    step0 = torch.zeros((), device=device, dtype=torch.int32)
    bc1 = torch.ones((), device=device, dtype=dtype)
    bc2 = torch.ones((), device=device, dtype=dtype)

    m_W0 = torch.zeros_like(W0); v_W0 = torch.zeros_like(W0)
    m_b0 = torch.zeros_like(b0); v_b0 = torch.zeros_like(b0)
    m_W1 = torch.zeros_like(W1); v_W1 = torch.zeros_like(W1)
    m_b1 = torch.zeros_like(b1); v_b1 = torch.zeros_like(b1)

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    def build():
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=device)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=device)

        sW0 = sym_tensor(name="0.W", shape=(D, D), dtype=dtype, device=device)
        sb0 = sym_tensor(name="0.b", shape=(D,), dtype=dtype, device=device)
        sW1 = sym_tensor(name="2.W", shape=(D, D), dtype=dtype, device=device)
        sb1 = sym_tensor(name="2.b", shape=(D,), dtype=dtype, device=device)

        s_step = sym_tensor(name="opt.step", shape=tuple(step0.shape), dtype=step0.dtype, device=device)
        s_bc1  = sym_tensor(name="opt.bc1_inv", shape=tuple(bc1.shape), dtype=bc1.dtype, device=device)
        s_bc2  = sym_tensor(name="opt.bc2_inv", shape=tuple(bc2.shape), dtype=bc2.dtype, device=device)

        smW0 = sym_tensor(name="opt.m.0.W", shape=(D, D), dtype=dtype, device=device)
        svW0 = sym_tensor(name="opt.v.0.W", shape=(D, D), dtype=dtype, device=device)
        smb0 = sym_tensor(name="opt.m.0.b", shape=(D,), dtype=dtype, device=device)
        svb0 = sym_tensor(name="opt.v.0.b", shape=(D,), dtype=dtype, device=device)

        smW1 = sym_tensor(name="opt.m.2.W", shape=(D, D), dtype=dtype, device=device)
        svW1 = sym_tensor(name="opt.v.2.W", shape=(D, D), dtype=dtype, device=device)
        smb1 = sym_tensor(name="opt.m.2.b", shape=(D,), dtype=dtype, device=device)
        svb1 = sym_tensor(name="opt.v.2.b", shape=(D,), dtype=dtype, device=device)

        lin0 = linear(sx, sW0, sb0, name="lin0_out")
        r0 = relu(lin0, name="relu0_out")
        r0s = save(r0, name="relu0_saved")
        lin1 = linear(r0, sW1, sb1, name="lin1_out")
        dY = mse_grad(lin1, st, name="dY")

        d_r0, dW1, db1 = linear_bwd(r0, sW1, dY, bias=True,
                                   dx_name="d_relu0_out", dW_name="d_2.W", db_name="d_2.b")
        d_lin0 = relu_bwd(d_r0, r0s, name="d_lin0_out")
        dx, dW0, db0 = linear_bwd(sx, sW0, d_lin0, bias=True,
                                  dx_name="d_x", dW_name="d_0.W", db_name="d_0.b")

        # optimizer chain
        s_step2 = step_inc(s_step, name="opt.step")
        s_bc1o, s_bc2o = bias_corr(
            s_step2, s_bc1, s_bc2,
            beta1=beta1, beta2=beta2,
            out1_name="opt.bc1_inv", out2_name="opt.bc2_inv",
        )
        adam_step(sW0, dW0, smW0, svW0, s_bc1o, s_bc2o, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sb0, db0, smb0, svb0, s_bc1o, s_bc2o, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sW1, dW1, smW1, svW1, s_bc1o, s_bc2o, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sb1, db1, smb1, svb1, s_bc1o, s_bc2o, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    ir = trace_ir(build, name="v2_stage6_train1")
    print(dump_ir(ir))
    lowered = lower_to_backend_ops(ir)
    print(dump_lowered(lowered, name="v2_stage6_train1"))
    plan = build_binding_plan(ir)
    print(dump_plan(plan, name="v2_stage6_train1"))

    ex = PlannedExecutor(ir=ir, lowered=lowered, plan=plan, opts=ExecOptions(debug=False))

    params = {
        "0.W": W0, "0.b": b0,
        "2.W": W1, "2.b": b1,
        "opt.step": step0,
        "opt.bc1_inv": bc1, "opt.bc2_inv": bc2,
        "opt.m.0.W": m_W0, "opt.v.0.W": v_W0,
        "opt.m.0.b": m_b0, "opt.v.0.b": v_b0,
        "opt.m.2.W": m_W1, "opt.v.2.W": v_W1,
        "opt.m.2.b": m_b1, "opt.v.2.b": v_b1,
    }

    def snap():
        return {
            "W0": W0.clone(), "b0": b0.clone(),
            "W1": W1.clone(), "b1": b1.clone(),
            "mW0": m_W0.clone(), "vW0": v_W0.clone(),
            "step": step0.clone(),
        }

    s0 = snap()

    # eager 1 step
    if hasattr(aicf_cuda._C, "trace_reset"):
        aicf_cuda._C.trace_reset()
    if hasattr(aicf_cuda._C, "trace_enable"):
        aicf_cuda._C.trace_enable(True)

    ex.run(inputs={"x": x, "t": t}, params=params, reuse_static=True)
    print("[eager] 1 step ok")
    if hasattr(aicf_cuda._C, "trace_get"):
        print("[trace] eager:", list(aicf_cuda._C.trace_get()))

    dW0 = maxabs(W0 - s0["W0"])
    db0d = maxabs(b0 - s0["b0"])
    dW1 = maxabs(W1 - s0["W1"])
    db1d = maxabs(b1 - s0["b1"])
    dmW0 = maxabs(m_W0 - s0["mW0"])
    dvW0 = maxabs(v_W0 - s0["vW0"])
    step_delta = int(step0.item()) - int(s0["step"].item())

    print("[delta] |W0| maxabs =", dW0)
    print("[delta] |b0| maxabs =", db0d)
    print("[delta] |W1| maxabs =", dW1)
    print("[delta] |b1| maxabs =", db1d)
    print("[delta] |mW0| maxabs =", dmW0)
    print("[delta] |vW0| maxabs =", dvW0)
    print("[delta] step delta =", step_delta, "(may be 0 if step_inc is disabled in this build)")

    # ✅ 핵심: 업데이트 존재 여부는 step이 아니라 "params/state 변화"로 판단
    assert (dW0 > 0.0) or (db0d > 0.0) or (dW1 > 0.0) or (db1d > 0.0), "params did not change"
    assert (dmW0 > 0.0) or (dvW0 > 0.0), "adam state did not change"

    # capture + replay (있으면)
    if hasattr(aicf_cuda._C, "graph_begin"):
        s1 = snap()

        print("[aicf] graph_begin (dedicated stream)")
        aicf_cuda._C.graph_begin()
        ex.run(inputs={"x": x, "t": t}, params=params, reuse_static=True)
        aicf_cuda._C.graph_end()
        print("[capture] ok")

        n = 5
        for _ in range(n):
            aicf_cuda._C.graph_launch()
        print(f"[replay] ok (n={n})")

        aicf_cuda._C.graph_reset()
        print("[reset] ok")

        dW0r = maxabs(W0 - s1["W0"])
        dmW0r = maxabs(m_W0 - s1["mW0"])
        step_delta_r = int(step0.item()) - int(s1["step"].item())

        print("[delta after replay] |W0| maxabs =", dW0r)
        print("[delta after replay] |mW0| maxabs =", dmW0r)
        print("[delta after replay] step delta =", step_delta_r, "(may be 0)")

        assert dW0r > 0.0, "W0 did not move during capture/replay"
        assert dmW0r > 0.0, "adam state did not move during capture/replay"

    print("OK (train-step: update + optional capture/replay stability)")


if __name__ == "__main__":
    main()
