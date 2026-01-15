from __future__ import annotations

import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.core_v2 import (
    trace_ir,
    dump_ir,
    dump_lowered,
    dump_plan,
)
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


def main():
    import aicf_fw.backend as BK
    from aicf_fw.backend.aicf_backend import AICFBackend  # 네 프로젝트 실제 backend 경로로 맞춰

    BK.set_backend(AICFBackend())
    
    torch.manual_seed(0)
    device = torch.device("cuda")
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

    # Adam states (must be pointer-stable)
    # step: int32 scalar on cuda
    step0 = torch.zeros((), device=device, dtype=torch.int32)
    bc1 = torch.ones((), device=device, dtype=dtype)
    bc2 = torch.ones((), device=device, dtype=dtype)

    m_W0 = torch.zeros_like(W0); v_W0 = torch.zeros_like(W0)
    m_b0 = torch.zeros_like(b0); v_b0 = torch.zeros_like(b0)
    m_W1 = torch.zeros_like(W1); v_W1 = torch.zeros_like(W1)
    m_b1 = torch.zeros_like(b1); v_b1 = torch.zeros_like(b1)

    # hyperparams
    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

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

        # optimizer states (param role)
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
        d_r0, dW1, db1 = linear_bwd(r0, sW1, dY, bias=True, dx_name="d_relu0_out", dW_name="d_2.W", db_name="d_2.b")
        d_lin0 = relu_bwd(d_r0, r0s, name="d_lin0_out")
        dx, dW0, db0 = linear_bwd(sx, sW0, d_lin0, bias=True, dx_name="d_x", dW_name="d_0.W", db_name="d_0.b")

        # optimizer: step/biascorr
        s_step2 = step_inc(s_step, name="opt.step")  # overwrite by name
        s_bc1o, s_bc2o = bias_corr(s_step2, s_bc1, s_bc2, beta1=beta1, beta2=beta2)

        # adam update (in-place semantic)
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

    # -------------------------
    # Execute via PlannedExecutor
    # -------------------------
    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        opts=ExecOptions(debug=False),
    )

    env = ex.run(
        inputs={"x": x, "t": t},
        params={
            "0.W": W0, "0.b": b0,
            "2.W": W1, "2.b": b1,
            "opt.step": step0,
            "opt.bc1_inv": bc1,
            "opt.bc2_inv": bc2,
            "opt.m.0.W": m_W0, "opt.v.0.W": v_W0,
            "opt.m.0.b": m_b0, "opt.v.0.b": v_b0,
            "opt.m.2.W": m_W1, "opt.v.2.W": v_W1,
            "opt.m.2.b": m_b1, "opt.v.2.b": v_b1,
        },
        reuse_static=True,
    )

    # -------------------------
    # Torch reference (same math)
    # -------------------------
    # forward
    lin0_t = x @ W0.t() + b0
    relu0_t = torch.relu(lin0_t)
    lin1_t = relu0_t @ W1.t() + b1

    # mse_grad
    dY_t = 2.0 * (lin1_t - t) / (B * D)  # if your mse_grad is mean over all elems (adjust if needed)

    # backward
    d_relu0_t = dY_t @ W1
    dW1_t = dY_t.t() @ relu0_t
    db1_t = dY_t.sum(dim=0)

    d_lin0_t = d_relu0_t * (relu0_t > 0).to(dtype)
    dW0_t = d_lin0_t.t() @ x
    db0_t = d_lin0_t.sum(dim=0)

    # adam ref (single step) - same formulas assumed
    # NOTE: 이 부분은 너의 adam_step 커널 정의와 정확히 동일해야 한다.
    #       (bias_corr 방식, bc1_inv/bc2_inv 정의 포함)
    with torch.no_grad():
        step_ref = step0.item()  # after execution step0 is updated in-place by backend
        # 여기서는 "결과 비교"를 최소화하려면 grad 비교 + weight가 변했는지만 체크해도 됨.

    # quick checks (grad correctness from env)
    # find vids by name
    name_to_vid = {v.name: int(vid) for vid, v in ir.values.items()}

    def maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
        return float((a - b).abs().max().item())

    print("[check] dW0 maxdiff =", maxdiff(env[name_to_vid["d_0.W"]], dW0_t))
    print("[check] db0 maxdiff =", maxdiff(env[name_to_vid["d_0.b"]], db0_t))
    print("[check] dW1 maxdiff =", maxdiff(env[name_to_vid["d_2.W"]], dW1_t))
    print("[check] db1 maxdiff =", maxdiff(env[name_to_vid["d_2.b"]], db1_t))

    print("OK (Stage6 skeleton)")

if __name__ == "__main__":
    main()
