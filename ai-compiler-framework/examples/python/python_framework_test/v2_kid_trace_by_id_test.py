from __future__ import annotations

import sys
from pathlib import Path
import time
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
    sgd_step,
)
from aicf_fw.core_v2.lower import lower_to_backend_ops
from aicf_fw.core_v2.plan import build_binding_plan, apply_kernel_decisions_stageB
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions


# -------------------------
# utils
# -------------------------
def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def require_contains(hay: str, needle: str):
    if needle not in hay:
        raise AssertionError(f"expected to find '{needle}' in trace, but not found.")


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


# -------------------------
# main
# -------------------------
def main():
    tf32_off()
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    dtype = torch.float16
    B, D = 64, 8  # vec2/half2 조건

    x = torch.randn(B, D, device=device, dtype=dtype)
    t = torch.randn(B, D, device=device, dtype=dtype)

    W0 = torch.randn(D, D, device=device, dtype=dtype)
    b0 = torch.randn(D, device=device, dtype=dtype)
    W1 = torch.randn(D, D, device=device, dtype=dtype)
    b1 = torch.randn(D, device=device, dtype=dtype)

    lr = 1e-2

    # -------------------------
    # build graph
    # -------------------------
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

    # -------------------------
    # artifacts dir
    # -------------------------
    art_root = ensure_dir(
        Path("artifacts") / f"{now_tag()}_v2_kid_trace_by_id_test"
    )

    # -------------------------
    # trace → IR
    # -------------------------
    ir = trace_ir(build, name="v2_kid_trace_by_id_test")
    ir_txt = dump_ir(ir)
    print(ir_txt)
    dump_text(art_root / "20_ir.txt", ir_txt)

    # -------------------------
    # lower Stage A + B
    # -------------------------
    lowered = lower_to_backend_ops(ir)                    # Stage A
    lowered = apply_kernel_decisions_stageB(ir, lowered)  # Stage B

    lowered_txt = dump_lowered(lowered, name="v2_kid_trace_by_id_test")
    print(lowered_txt)
    dump_text(art_root / "30_lowered.txt", lowered_txt)

    # kernel_id 전체 존재 확인
    kids = [it.get("kernel_id", None) for it in lowered]
    if any(k is None for k in kids):
        missing = [i for i, k in enumerate(kids) if k is None]
        raise RuntimeError(f"Some lowered ops have no kernel_id. missing indices={missing}")

    # -------------------------
    # plan
    # -------------------------
    plan = build_binding_plan(ir)
    plan_txt = dump_plan(plan, name="v2_kid_trace_by_id_test")
    print(plan_txt)
    dump_text(art_root / "40_plan.txt", plan_txt)

    # -------------------------
    # exec + runtime trace
    # -------------------------
    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        opts=ExecOptions(debug=False, require_kernel_id=True),
    )

    params = {"0.W": W0, "0.b": b0, "2.W": W1, "2.b": b1}

    ex.trace_reset()
    ex.trace_enable(True)
    ex.run(inputs={"x": x, "t": t}, params=params, reuse_static=True)
    ex.trace_enable(False)

    lines = ex.trace_get()
    if not lines:
        raise RuntimeError("trace_get() returned empty. Check C++ trace implementation is enabled.")

    trace = "\n".join([str(s) for s in lines])
    print("=== TRACE ===")
    print(trace)
    dump_text(art_root / "50_runtime_trace.txt", trace)

    # 최소 기대: stageB 업그레이드된 kid가 실제 호출되었는지
    require_contains(trace, "sgd_step_f16_half2_v0")

    print(f"[OK] artifacts dumped to: {art_root}")
    print("OK")


if __name__ == "__main__":
    main()
