# examples/python/python_framework_test/v2_stage6_train1_check_grad_only_hostmeta.py
from __future__ import annotations

import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.core_v2 import trace_ir, dump_ir, dump_lowered, dump_plan
from aicf_fw.core_v2.ops import (
    sym_tensor,
    linear,
    relu,
    save,
    mse_grad,
    linear_bwd,
    relu_bwd,
    adam_step,
)
from aicf_fw.core_v2.lower import lower_to_backend_ops
from aicf_fw.core_v2.plan import build_binding_plan
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def maxabs(a: torch.Tensor) -> float:
    return float(a.abs().max().item())


def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def main():
    tf32_off()
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    dtype = torch.float32
    B, D = 64, 8

    # runtime inputs
    x = torch.randn(B, D, device=device, dtype=dtype)
    t = torch.randn(B, D, device=device, dtype=dtype)

    # params
    W0 = torch.randn(D, D, device=device, dtype=dtype)
    b0 = torch.randn(D, device=device, dtype=dtype)
    W1 = torch.randn(D, D, device=device, dtype=dtype)
    b1 = torch.randn(D, device=device, dtype=dtype)

    # optimizer state
    m_W0 = torch.zeros_like(W0); v_W0 = torch.zeros_like(W0)
    m_b0 = torch.zeros_like(b0); v_b0 = torch.zeros_like(b0)
    m_W1 = torch.zeros_like(W1); v_W1 = torch.zeros_like(W1)
    m_b1 = torch.zeros_like(b1); v_b1 = torch.zeros_like(b1)

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    # host-managed meta (fixed tensors; values updated by fill_)
    bc1_inv = torch.ones((), device=device, dtype=dtype)
    bc2_inv = torch.ones((), device=device, dtype=dtype)

    def hostmeta_set_step(step: int):
        # NOTE: your kernel expects "inv bias-corr terms" (actually multiplicative correction factors)
        bc1 = 1.0 / (1.0 - (beta1 ** step))
        bc2 = 1.0 / (1.0 - (beta2 ** step))
        bc1_inv.fill_(float(bc1))
        bc2_inv.fill_(float(bc2))

    def build():
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=device)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=device)

        sW0 = sym_tensor(name="0.W", shape=(D, D), dtype=dtype, device=device)
        sb0 = sym_tensor(name="0.b", shape=(D,), dtype=dtype, device=device)
        sW1 = sym_tensor(name="2.W", shape=(D, D), dtype=dtype, device=device)
        sb1 = sym_tensor(name="2.b", shape=(D,), dtype=dtype, device=device)

        # ✅ host-managed optimizer meta: no step_inc / no bias_corr
        s_bc1 = sym_tensor(name="opt.bc1_inv", shape=(), dtype=dtype, device=device)
        s_bc2 = sym_tensor(name="opt.bc2_inv", shape=(), dtype=dtype, device=device)

        smW0 = sym_tensor(name="opt.m.0.W", shape=(D, D), dtype=dtype, device=device)
        svW0 = sym_tensor(name="opt.v.0.W", shape=(D, D), dtype=dtype, device=device)
        smb0 = sym_tensor(name="opt.m.0.b", shape=(D,), dtype=dtype, device=device)
        svb0 = sym_tensor(name="opt.v.0.b", shape=(D,), dtype=dtype, device=device)

        smW1 = sym_tensor(name="opt.m.2.W", shape=(D, D), dtype=dtype, device=device)
        svW1 = sym_tensor(name="opt.v.2.W", shape=(D, D), dtype=dtype, device=device)
        smb1 = sym_tensor(name="opt.m.2.b", shape=(D,), dtype=dtype, device=device)
        svb1 = sym_tensor(name="opt.v.2.b", shape=(D,), dtype=dtype, device=device)

        # forward
        lin0 = linear(sx, sW0, sb0, name="lin0_out")
        r0 = relu(lin0, name="relu0_out")
        r0s = save(r0, name="relu0_saved")
        lin1 = linear(r0, sW1, sb1, name="lin1_out")
        dY = mse_grad(lin1, st, name="dY")

        # backward
        d_r0, dW1, db1 = linear_bwd(
            r0, sW1, dY,
            bias=True,
            dx_name="d_relu0_out",
            dW_name="d_2.W",
            db_name="d_2.b",
        )
        d_lin0 = relu_bwd(d_r0, r0s, name="d_lin0_out")
        dx, dW0, db0 = linear_bwd(
            sx, sW0, d_lin0,
            bias=True,
            dx_name="d_x",
            dW_name="d_0.W",
            db_name="d_0.b",
        )

        # adam (host-provided bc1/bc2)
        adam_step(sW0, dW0, smW0, svW0, s_bc1, s_bc2, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sb0, db0, smb0, svb0, s_bc1, s_bc2, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sW1, dW1, smW1, svW1, s_bc1, s_bc2, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sb1, db1, smb1, svb1, s_bc1, s_bc2, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    ir = trace_ir(build, name="v2_stage6_train1_hostmeta")
    print(dump_ir(ir))

    lowered = lower_to_backend_ops(ir)
    print(dump_lowered(lowered, name="v2_stage6_train1_hostmeta"))

    plan = build_binding_plan(ir)
    print(dump_plan(plan, name="v2_stage6_train1_hostmeta"))

    ex = PlannedExecutor(ir=ir, lowered=lowered, plan=plan, opts=ExecOptions(debug=False))

    params = {
        "0.W": W0, "0.b": b0,
        "2.W": W1, "2.b": b1,
        "opt.bc1_inv": bc1_inv,
        "opt.bc2_inv": bc2_inv,
        "opt.m.0.W": m_W0, "opt.v.0.W": v_W0,
        "opt.m.0.b": m_b0, "opt.v.0.b": v_b0,
        "opt.m.2.W": m_W1, "opt.v.2.W": v_W1,
        "opt.m.2.b": m_b1, "opt.v.2.b": v_b1,
    }

    # --- hostmeta step=1 ---
    hostmeta_set_step(1)
    print("[meta] step=1 bc1_inv=", float(bc1_inv.item()), "bc2_inv=", float(bc2_inv.item()))

    # snapshot for update sanity (even though script is "grad-only", this catches the no-op p-update bug)
    W0_0 = W0.clone()
    mW0_0 = m_W0.clone()
    vW0_0 = v_W0.clone()

    # warmup 1
    env = ex.run(inputs={"x": x, "t": t}, params=params, reuse_static=True)
    print("[run] ok")
    print("[trace] eager:", ex.trace_get() if hasattr(ex, "trace_get") else "(no trace_get)")

    # ---- binding sanity: env vid for "0.W" must alias W0 ----
    name_to_vid = {v.name: int(vid) for vid, v in ir.values.items()}
    vW0 = name_to_vid["0.W"]
    same_ptr = int(env[vW0].data_ptr()) == int(W0.data_ptr())
    print("[bind] env['0.W'] ptr =", env[vW0].data_ptr(), "W0 ptr =", W0.data_ptr(), "same_ptr =", same_ptr)
    if not same_ptr:
        raise RuntimeError("Binding mismatch: env['0.W'] is not the same tensor as W0 (plan alias broken).")

    # torch ref grads (same layout: y = x @ W^T + b)
    lin0_t = x @ W0.t() + b0
    relu0_t = torch.relu(lin0_t)
    lin1_t = relu0_t @ W1.t() + b1

    dY_t = 2.0 * (lin1_t - t) / (B * D)
    d_relu0_t = dY_t @ W1
    dW1_t = dY_t.t() @ relu0_t
    db1_t = dY_t.sum(dim=0)
    d_lin0_t = d_relu0_t * (relu0_t > 0).to(dtype)
    dW0_t = d_lin0_t.t() @ x
    db0_t = d_lin0_t.sum(dim=0)

    # grad checks
    print("[check] dW0 maxdiff =", maxdiff(env[name_to_vid["d_0.W"]], dW0_t))
    print("[check] db0 maxdiff =", maxdiff(env[name_to_vid["d_0.b"]], db0_t))
    print("[check] dW1 maxdiff =", maxdiff(env[name_to_vid["d_2.W"]], dW1_t))
    print("[check] db1 maxdiff =", maxdiff(env[name_to_vid["d_2.b"]], db1_t))
    print("[check] dY  maxdiff =", maxdiff(env[name_to_vid["dY"]], dY_t))
    print("[check] d_lin0 maxdiff =", maxdiff(env[name_to_vid["d_lin0_out"]], d_lin0_t))

    # update sanity (minimal)
    dW0_param = maxdiff(W0, W0_0)
    dM0 = maxdiff(m_W0, mW0_0)
    dV0 = maxdiff(v_W0, vW0_0)
    print("[delta] |W0|", dW0_param, "|mW0|", dM0, "|vW0|", dV0)
    print("[grad]  |dW0|maxabs =", maxabs(env[name_to_vid["d_0.W"]]))

    # if state changes but param doesn't => classic "p-update path disabled" symptom
    if dM0 > 0.0 and dW0_param == 0.0:
        raise RuntimeError("adam_step updated m/v but NOT p (W0). p-update path likely no-op/disabled.")
    if dM0 == 0.0 and dW0_param == 0.0:
        raise RuntimeError("Neither param nor state changed. adam_step likely not executed or outputs not wired.")

    print("OK (grad + minimal update sanity, host-managed meta)")


if __name__ == "__main__":
    main()
