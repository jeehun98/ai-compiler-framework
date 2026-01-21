from __future__ import annotations

import torch

from _test_path_bootstrap import ensure_test_paths

ensure_test_paths()

from aicf_fw.core_v2 import trace_ir, dump_ir, dump_lowered, dump_plan
from aicf_fw.core_v2.ops import (
    sym_tensor,
    linear,
    relu,
    save,
    mse_grad,
    linear_bwd,
    relu_bwd,
    sgd_step,
)
from aicf_fw.core_v2.lower import lower_to_backend_ops
from aicf_fw.core_v2.plan import build_binding_plan, apply_kernel_decisions_stageB


def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def build_graph(B: int, D: int, dtype: torch.dtype, device: torch.device, lr: float):
    def build():
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=device)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=device)

        sW0 = sym_tensor(name="0.W", shape=(D, D), dtype=dtype, device=device)
        sb0 = sym_tensor(name="0.b", shape=(D,), dtype=dtype, device=device)
        sW1 = sym_tensor(name="2.W", shape=(D, D), dtype=dtype, device=device)
        sb1 = sym_tensor(name="2.b", shape=(D,), dtype=dtype, device=device)

        lin0 = linear(sx, sW0, sb0, name="lin0_out")
        r0 = relu(lin0, name="relu0_out")
        r0s = save(r0, name="relu0_saved")
        lin1 = linear(r0, sW1, sb1, name="lin1_out")
        dY = mse_grad(lin1, st, name="dY")

        d_r0, dW1, db1 = linear_bwd(
            r0, sW1, dY,
            bias=True,
            dx_name="d_relu0_out",
            dW_name="d_2.W",
            db_name="d_2.b",
        )
        d_lin0 = relu_bwd(d_r0, r0s, name="d_lin0_out")
        _dx, dW0, db0 = linear_bwd(
            sx, sW0, d_lin0,
            bias=True,
            dx_name="d_x",
            dW_name="d_0.W",
            db_name="d_0.b",
        )

        sgd_step(sW0, dW0, lr=lr)
        sgd_step(sb0, db0, lr=lr)
        sgd_step(sW1, dW1, lr=lr)
        sgd_step(sb1, db1, lr=lr)

    ir = trace_ir(build, name="v2_kid_stageAB_test")
    lowered_a = lower_to_backend_ops(ir)                    # Stage A
    lowered_b = apply_kernel_decisions_stageB(ir, lowered_a)  # Stage B
    plan = build_binding_plan(ir)
    return ir, lowered_a, lowered_b, plan


def main():
    tf32_off()
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    dtype = torch.float16
    B, D = 64, 8
    lr = 1e-2

    ir, lowered_a, lowered_b, plan = build_graph(B, D, dtype, device, lr)

    print(dump_ir(ir))
    print("=== LoweredOps(StageA) ===")
    print(dump_lowered(lowered_a, name="v2_kid_stageAB_test_stageA"))
    print("=== LoweredOps(StageB) ===")
    print(dump_lowered(lowered_b, name="v2_kid_stageAB_test_stageB"))
    print(dump_plan(plan, name="v2_kid_stageAB_test"))

    # kernel_id 존재성
    kids = [it.get("kernel_id", None) for it in lowered_b]
    if any(k is None for k in kids):
        missing = [i for i, k in enumerate(kids) if k is None]
        raise RuntimeError(f"Some lowered ops have no kernel_id. missing indices={missing}")

    ks = [str(k) for k in kids]
    has_vec2 = any("vec2" in k for k in ks)
    has_half2 = any("half2" in k for k in ks)

    if not has_vec2:
        print("[warn] no vec2 kernel_id found (maybe conditions not met).")
    if not has_half2:
        raise RuntimeError("Expected at least one half2 kernel_id, but none found. (check sgd_step upgrade)")

    print("[OK] StageA/StageB: kernel_id present; vec2/half2 decisions applied (half2 required).")
    print("OK")


if __name__ == "__main__":
    main()
