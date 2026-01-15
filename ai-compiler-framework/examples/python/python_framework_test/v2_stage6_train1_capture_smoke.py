# examples/python/python_framework_test/v2_stage6_train1_capture_smoke.py
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
    linear, relu, save,
    mse_grad,
    linear_bwd, relu_bwd,
    step_inc, bias_corr, adam_step,
)
from aicf_fw.core_v2.lower import lower_to_backend_ops
from aicf_fw.core_v2.plan import build_binding_plan
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions


def ptr(x: torch.Tensor) -> int:
    return int(x.untyped_storage().data_ptr())


def main():
    import aicf_fw.backend as BK
    from aicf_fw.backend.aicf_backend import AICFBackend
    BK.set_backend(AICFBackend())

    # ---- deterministic-ish (TF32 off) ----
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    device = torch.device("cuda:0")
    dtype = torch.float32
    B, D = 64, 8

    # -------------------------
    # 고정 파라미터/옵티마 상태 (포인터 고정)
    # -------------------------
    W0 = torch.randn(D, D, device=device, dtype=dtype)
    b0 = torch.randn(D, device=device, dtype=dtype)
    W1 = torch.randn(D, D, device=device, dtype=dtype)
    b1 = torch.randn(D, device=device, dtype=dtype)

    step0 = torch.zeros((), device=device, dtype=torch.int32)

    m_W0 = torch.zeros_like(W0); v_W0 = torch.zeros_like(W0)
    m_b0 = torch.zeros_like(b0); v_b0 = torch.zeros_like(b0)
    m_W1 = torch.zeros_like(W1); v_W1 = torch.zeros_like(W1)
    m_b1 = torch.zeros_like(b1); v_b1 = torch.zeros_like(b1)

    lr = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps = 1e-8

    # -------------------------
    # IR build
    # -------------------------
    def build():
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=device)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=device)

        sW0 = sym_tensor(name="0.W", shape=(D, D), dtype=dtype, device=device)
        sb0 = sym_tensor(name="0.b", shape=(D,), dtype=dtype, device=device)
        sW1 = sym_tensor(name="2.W", shape=(D, D), dtype=dtype, device=device)
        sb1 = sym_tensor(name="2.b", shape=(D,), dtype=dtype, device=device)

        s_step = sym_tensor(name="opt.step", shape=tuple(step0.shape), dtype=step0.dtype, device=device)

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

        # optimizer
        s_step2 = step_inc(s_step, name="opt.step")

        bc1o, bc2o = bias_corr(
            s_step2,
            sym_tensor(name="__dummy_bc1__", shape=(), dtype=dtype, device=device),
            sym_tensor(name="__dummy_bc2__", shape=(), dtype=dtype, device=device),
            beta1=beta1, beta2=beta2
        )

        adam_step(sW0, dW0, smW0, svW0, bc1o, bc2o, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sb0, db0, smb0, svb0, bc1o, bc2o, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sW1, dW1, smW1, svW1, bc1o, bc2o, lr=lr, beta1=beta1, beta2=beta2, eps=eps)
        adam_step(sb1, db1, smb1, svb1, bc1o, bc2o, lr=lr, beta1=beta1, beta2=beta2, eps=eps)

        _ = dx  # keep alive

    ir = trace_ir(build, name="v2_stage6_train1")
    lowered = lower_to_backend_ops(ir)
    plan = build_binding_plan(ir)

    # 로그는 필요하면 켜고, 캡처 시엔 웬만하면 꺼두는게 안전
    print(dump_ir(ir))
    print(dump_lowered(lowered, title="LoweredOps(v2_stage6_train1)"))
    print(dump_plan(plan, title="BindingPlan(v2_stage6_train1)"))

    ex = PlannedExecutor(
        ir=ir, lowered=lowered, plan=plan,
        opts=ExecOptions(debug=False),
        device=device,
    )

    # -------------------------
    # ✅ 입력 버퍼 고정 (여기에 매 step copy_in)
    # -------------------------
    x_buf = torch.empty((B, D), device=device, dtype=dtype)
    t_buf = torch.empty((B, D), device=device, dtype=dtype)

    # params dict도 캡처/리플레이 내내 동일 객체로 유지
    params = {
        "0.W": W0, "0.b": b0,
        "2.W": W1, "2.b": b1,
        "opt.step": step0,
        "opt.m.0.W": m_W0, "opt.v.0.W": v_W0,
        "opt.m.0.b": m_b0, "opt.v.0.b": v_b0,
        "opt.m.2.W": m_W1, "opt.v.2.W": v_W1,
        "opt.m.2.b": m_b1, "opt.v.2.b": v_b1,
    }

    # 캡처용 step fn (입/파/상태 텐서 포인터가 바뀌면 안됨)
    def step_once():
        ex.run(
            inputs={"x": x_buf, "t": t_buf},
            params=params,
            reuse_static=True,
        )

    # -------------------------
    # warmup: plan/alloc 안정화
    # -------------------------
    for _ in range(3):
        x_buf.copy_(torch.randn((B, D), device=device, dtype=dtype))
        t_buf.copy_(torch.randn((B, D), device=device, dtype=dtype))
        step_once()
    torch.cuda.synchronize()

    # 포인터 스냅샷 (선택: 캡처 후에도 변하는지 체크)
    snap = {
        "x_buf": ptr(x_buf),
        "t_buf": ptr(t_buf),
        "W0": ptr(W0), "b0": ptr(b0), "W1": ptr(W1), "b1": ptr(b1),
        "step": ptr(step0),
        "m_W0": ptr(m_W0), "v_W0": ptr(v_W0),
        "m_b0": ptr(m_b0), "v_b0": ptr(v_b0),
        "m_W1": ptr(m_W1), "v_W1": ptr(v_W1),
        "m_b1": ptr(m_b1), "v_b1": ptr(v_b1),
    }

    # -------------------------
    # CUDA Graph Capture
    # -------------------------
    g = torch.cuda.CUDAGraph()
    # 선택: 메모리 풀 고정. (안 써도 되지만 안정성↑)
    pool = torch.cuda.graphs.graph_pool_handle()

    torch.cuda.synchronize()
    with torch.cuda.graph(g, pool=pool):
        step_once()
    torch.cuda.synchronize()

    # -------------------------
    # replay loop
    # -------------------------
    replay_n = 50
    for i in range(replay_n):
        # 입력값만 갱신 (포인터는 그대로)
        x_buf.copy_(torch.randn((B, D), device=device, dtype=dtype))
        t_buf.copy_(torch.randn((B, D), device=device, dtype=dtype))
        g.replay()

    torch.cuda.synchronize()

    # -------------------------
    # ptr invariant check (필수)
    # -------------------------
    for k, p0 in snap.items():
        p1 = ptr(eval(k)) if k.endswith("_buf") else ptr(eval(k))  # same names in locals
        if p0 != p1:
            raise AssertionError(f"[PTR] {k} changed: {p0} -> {p1}")

    print(f"OK (CUDA Graph capture+replay) replay_n={replay_n}")


if __name__ == "__main__":
    main()
