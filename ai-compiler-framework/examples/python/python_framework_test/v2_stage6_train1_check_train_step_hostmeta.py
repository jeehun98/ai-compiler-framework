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


def maxabs(a: torch.Tensor) -> float:
    return float(a.abs().max().item())


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


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

    # ✅ host-managed meta tensors (fixed pointers)
    bc1_inv = torch.ones((), device=device, dtype=dtype)
    bc2_inv = torch.ones((), device=device, dtype=dtype)
    step_host = 0

    def update_opt_meta():
        nonlocal step_host
        step_host += 1
        bc1 = 1.0 / (1.0 - (beta1 ** step_host))
        bc2 = 1.0 / (1.0 - (beta2 ** step_host))
        bc1_inv.fill_(float(bc1))
        bc2_inv.fill_(float(bc2))

    def build():
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=device)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=device)

        sW0 = sym_tensor(name="0.W", shape=(D, D), dtype=dtype, device=device)
        sb0 = sym_tensor(name="0.b", shape=(D,), dtype=dtype, device=device)
        sW1 = sym_tensor(name="2.W", shape=(D, D), dtype=dtype, device=device)
        sb1 = sym_tensor(name="2.b", shape=(D,), dtype=dtype, device=device)

        # host-managed meta only
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

        # adam (in-place overwrite pattern)
        adam_step(sW0, dW0, smW0, svW0, s_bc1, s_bc2, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sb0, db0, smb0, svb0, s_bc1, s_bc2, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sW1, dW1, smW1, svW1, s_bc1, s_bc2, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sb1, db1, smb1, svb1, s_bc1, s_bc2, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    # build IR
    ir = trace_ir(build, name="v2_stage6_train1_hostmeta")
    lowered = lower_to_backend_ops(ir)
    plan = build_binding_plan(ir)

    print(dump_ir(ir))
    print(dump_lowered(lowered, name="v2_stage6_train1_hostmeta"))
    print(dump_plan(plan, name="v2_stage6_train1_hostmeta"))

    ex = PlannedExecutor(ir=ir, lowered=lowered, plan=plan, opts=ExecOptions(debug=False))

    # --- bind params ---
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

    # snapshot before
    W0_0 = W0.clone()
    mW0_0 = m_W0.clone()
    vW0_0 = v_W0.clone()

    # --- eager 1 step ---
    update_opt_meta()
    print("[meta] step_host =", step_host, "bc1_inv=", float(bc1_inv.item()), "bc2_inv=", float(bc2_inv.item()))
    env = ex.run(inputs={"x": x, "t": t}, params=params, reuse_static=True)
    print("[eager] 1 step ok")

    # trace
    if hasattr(ex, "trace_get"):
        tr = ex.trace_get()
        print("[trace] eager:", tr[:30], ("...(+%d)" % (len(tr) - 30) if len(tr) > 30 else ""))

    # --- locate vids ---
    name_to_vid = {v.name: int(vid) for vid, v in ir.values.items()}

    # grads must exist in env
    dW0 = env[name_to_vid["d_0.W"]]
    db0 = env[name_to_vid["d_0.b"]]
    dW1 = env[name_to_vid["d_2.W"]]
    db1 = env[name_to_vid["d_2.b"]]

    print("[grad] |dW0| maxabs =", maxabs(dW0), " |db0| maxabs =", maxabs(db0))
    print("[grad] |dW1| maxabs =", maxabs(dW1), " |db1| maxabs =", maxabs(db1))

    # deltas (param + state)
    dW0_param = maxabs_delta(W0, W0_0)
    dM0 = maxabs_delta(m_W0, mW0_0)
    dV0 = maxabs_delta(v_W0, vW0_0)

    print("[delta] |W0| maxabs =", dW0_param)
    print("[delta] |mW0| maxabs =", dM0)
    print("[delta] |vW0| maxabs =", dV0)

    # ✅ binding sanity: env vid for "0.W" should be EXACT same storage as W0
    vW0 = name_to_vid["0.W"]
    same_ptr = int(env[vW0].data_ptr()) == int(W0.data_ptr())
    print("[bind] env['0.W'] ptr =", env[vW0].data_ptr(), "W0 ptr =", W0.data_ptr(), "same_ptr =", same_ptr)

    # ---- decisive branching ----
    if not same_ptr:
        raise RuntimeError(
            "W0 binding is NOT pointing to actual param tensor. "
            "Your plan/env is updating a different tensor than W0."
        )

    if maxabs(dW0) == 0.0 and maxabs(db0) == 0.0 and maxabs(dW1) == 0.0 and maxabs(db1) == 0.0:
        raise RuntimeError("All grads are exactly 0. Your forward/backward chain produced zero gradients.")

    # if state changes but param doesn't => adam_step is not updating p
    if dM0 > 0.0 and dW0_param == 0.0:
        raise RuntimeError(
            "adam_step updated m/v but NOT p. "
            "This strongly suggests your kernel's p-update path is disabled/no-op under these attrs/meta."
        )

    # if nothing changes => adam_step not running or output not written
    if dM0 == 0.0 and dW0_param == 0.0:
        raise RuntimeError(
            "Neither params nor optimizer state changed. "
            "Either adam_step didn't run (NotImplemented/early-exit) or outputs are not wired to in-place tensors."
        )

    # normal expectation: param update must happen
    if dW0_param == 0.0:
        raise AssertionError("params did not update (W0 delta == 0)")

    print("OK (train-step hostmeta: eager update works)")

    # --- capture/replay (optional) ---
    if hasattr(ex, "reset_graph"):
        ex.reset_graph()

    ex.capture(inputs={"x": x, "t": t}, params=params, reuse_static=True)
    print("[capture] ok")

    ex.replay(n=5)
    print("[replay] ok (n=5)")

    if hasattr(ex, "reset_graph"):
        ex.reset_graph()
        print("[reset] ok")

    print("OK (train-step hostmeta + capture/replay stability)")


if __name__ == "__main__":
    main()
