# examples/python/python_framework_test/v2_stage6_train1_check_skeleton.py
from __future__ import annotations

import os
import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
ROOT = THIS.parents[3]         # repo root
BUILD_PY = ROOT / "build" / "python"

# --- auto PYTHONPATH bootstrap (aicf_fw + aicf_cuda) ---
for p in (EXAMPLES_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
# -------------------------------------------------------

from aicf_fw.core_v2 import trace_ir, dump_ir, dump_lowered, dump_plan
from aicf_fw.core_v2.ops import (
    sym_tensor,
    linear,
    relu,
    save,
    mse_grad,
    linear_bwd,
    relu_bwd,
    step_inc,
    bias_corr,
    adam_step,
)
from aicf_fw.core_v2.lower import lower_to_backend_ops
from aicf_fw.core_v2.plan import build_binding_plan
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions


def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def main():
    tf32_off()

    # ✅ 중요: replay 포함 비교는 파라미터가 바뀌면 의미가 없어짐.
    # warmup=1이면 step_inc는 NO-OP, adam_step은 lr=0으로 업데이트 방지.
    os.environ.setdefault("AICF_WARMUP", "1")

    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.float32

    B, D = 64, 8

    # -------------------------
    # Runtime tensors (inputs/params/state)
    # -------------------------
    x = torch.randn(B, D, device=device, dtype=dtype)
    t = torch.randn(B, D, device=device, dtype=dtype)

    W0 = torch.randn(D, D, device=device, dtype=dtype)
    b0 = torch.randn(D, device=device, dtype=dtype)
    W1 = torch.randn(D, D, device=device, dtype=dtype)
    b1 = torch.randn(D, device=device, dtype=dtype)

    # optimizer states (pointer-stable)
    step0 = torch.zeros((), device=device, dtype=torch.int32)
    bc1 = torch.ones((), device=device, dtype=dtype)
    bc2 = torch.ones((), device=device, dtype=dtype)

    m_W0 = torch.zeros_like(W0); v_W0 = torch.zeros_like(W0)
    m_b0 = torch.zeros_like(b0); v_b0 = torch.zeros_like(b0)
    m_W1 = torch.zeros_like(W1); v_W1 = torch.zeros_like(W1)
    m_b1 = torch.zeros_like(b1); v_b1 = torch.zeros_like(b1)

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    # -------------------------
    # IR build (Stage6)
    # -------------------------
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
        _, dW0, db0 = linear_bwd(
            sx, sW0, d_lin0,
            bias=True,
            dx_name="d_x",
            dW_name="d_0.W",
            db_name="d_0.b",
        )

        # step update (overwrite name)
        s_step2 = step_inc(s_step, name="opt.step")

        # bias corr: overwrite bc storages for plan cleanliness
        s_bc1o, s_bc2o = bias_corr(
            s_step2, s_bc1, s_bc2,
            beta1=beta1, beta2=beta2,
            out1_name="opt.bc1_inv",
            out2_name="opt.bc2_inv",
        )

        # adam (in-place semantic)
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

    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        opts=ExecOptions(debug=False),
    )
    ex.trace_reset()
    ex.trace_enable(True)

    params = {
        "0.W": W0, "0.b": b0,
        "2.W": W1, "2.b": b1,
        "opt.step": step0,
        "opt.bc1_inv": bc1,
        "opt.bc2_inv": bc2,
        "opt.m.0.W": m_W0, "opt.v.0.W": v_W0,
        "opt.m.0.b": m_b0, "opt.v.0.b": v_b0,
        "opt.m.2.W": m_W1, "opt.v.2.W": v_W1,
        "opt.m.2.b": m_b1, "opt.v.2.b": v_b1,
    }

    # -------------------------
    # Eager run (single pass)
    # -------------------------
    env = ex.run(inputs={"x": x, "t": t}, params=params, reuse_static=True)
    print("[run] ok")
    print("[trace] eager:", ex.trace_get())

    # -------------------------
    # Capture + replay
    # -------------------------
    ex.trace_reset()
    ex.capture(inputs={"x": x, "t": t}, params=params, reuse_static=True)
    print("[capture] ok")
    print("[trace] capture:", ex.trace_get())

    ex.replay(n=5)
    print("[replay] ok (n=5)")

    ex.reset_graph()
    print("[reset] ok")

    # -------------------------
    # Torch reference (same weights; warmup keeps weights unchanged)
    # -------------------------
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

    name_to_vid = {v.name: int(vid) for vid, v in ir.values.items()}

    dY_env = env[name_to_vid["dY"]]
    dW0_env = env[name_to_vid["d_0.W"]]
    db0_env = env[name_to_vid["d_0.b"]]
    dW1_env = env[name_to_vid["d_2.W"]]
    db1_env = env[name_to_vid["d_2.b"]]
    dlin0_env = env[name_to_vid["d_lin0_out"]]

    print("[check] dW0 maxdiff =", maxdiff(dW0_env, dW0_t))
    print("[check] db0 maxdiff =", maxdiff(db0_env, db0_t))
    print("[check] dW1 maxdiff =", maxdiff(dW1_env, dW1_t))
    print("[check] db1 maxdiff =", maxdiff(db1_env, db1_t))
    print("[check] dY  maxdiff =", maxdiff(dY_env, dY_t))
    print("[check] d_lin0 maxdiff =", maxdiff(dlin0_env, d_lin0_t))

    atol_grad = 1e-4
    assert maxdiff(dW0_env, dW0_t) <= atol_grad
    assert maxdiff(db0_env, db0_t) <= atol_grad
    assert maxdiff(dW1_env, dW1_t) <= atol_grad
    assert maxdiff(db1_env, db1_t) <= atol_grad
    assert maxdiff(dlin0_env, d_lin0_t) <= atol_grad
    assert maxdiff(dY_env, dY_t) <= 5e-4  # dY is usually slightly looser

    print("OK (Stage6 skeleton + capture/replay, warmup=1)")

if __name__ == "__main__":
    main()
