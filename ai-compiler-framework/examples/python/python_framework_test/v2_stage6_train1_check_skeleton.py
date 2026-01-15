# examples/python/python_framework_test/v2_stage6_train1_check_skeleton.py
from __future__ import annotations

import sys
from pathlib import Path
import torch

# --- PYTHONPATH bootstrap (examples/python) ---
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))
# --------------------------------------------

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


# -------------------------
# small utils
# -------------------------
def mdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float):
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        raise AssertionError(f"[FAIL] {name}: maxdiff={mdiff(got, ref)} atol={atol} rtol={rtol}")


def ptr(x: torch.Tensor) -> int:
    return int(x.untyped_storage().data_ptr())


def _is_finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


# -------------------------
# invariants / diagnostics
# -------------------------
def check_env_invariants(ir, env: dict[int, torch.Tensor], *, device: torch.device):
    """
    강제:
      - device
      - shape (ir.value.shape가 있으면)
    경고(FAIL X):
      - dtype (IR dtype 표현체가 torch.dtype이 아닐 수 있어서 string 비교만)
    진단:
      - NaN/Inf
      - storage ptr alias group 출력
    """
    dtype_warn = 0
    nonfinite = 0

    for vid, v in ir.values.items():
        vid = int(vid)
        t = env.get(vid, None)
        if t is None:
            continue

        # device strict
        if t.device != device:
            raise AssertionError(
                f"[INV] vid={vid} name={v.name}: device mismatch: got={t.device} expected={device}"
            )

        # shape strict (if available)
        if hasattr(v, "shape"):
            if tuple(t.shape) != tuple(v.shape):
                raise AssertionError(
                    f"[INV] vid={vid} name={v.name}: shape mismatch: got={tuple(t.shape)} expected={tuple(v.shape)}"
                )

        # dtype warn-only (string compare)
        if hasattr(v, "dtype"):
            if str(t.dtype) != str(v.dtype):
                dtype_warn += 1
                print(f"[WARN][dtype] vid={vid} name={v.name}: got={t.dtype} expected={v.dtype}")

        # finite check
        if not _is_finite(t):
            nonfinite += 1
            print(f"[WARN][finite] vid={vid} name={v.name}: contains NaN/Inf")

    # ptr alias diagnosis
    ptr_map: dict[int, list[tuple[int, str]]] = {}
    for vid, v in ir.values.items():
        vid = int(vid)
        t = env.get(vid, None)
        if t is None:
            continue
        p = ptr(t)
        ptr_map.setdefault(p, []).append((vid, v.name))

    alias_groups = [items for items in ptr_map.values() if len(items) >= 2]
    if alias_groups:
        print("\n[diag] potential alias groups (same storage ptr across multiple values):")
        for g in alias_groups:
            print("  - " + ", ".join([f"{vid}:{nm}" for vid, nm in g]))
        print("[diag] (If unintended, can cause overwrite bugs when reuse_static=True.)\n")

    if dtype_warn:
        print(f"[diag] dtype warnings = {dtype_warn} (non-fatal)")
    if nonfinite:
        raise AssertionError(f"[INV] non-finite tensors detected: count={nonfinite}")


@torch.no_grad()
def torch_ref_grads_cpu_fp64(x, t, W0, b0, W1, b1):
    """
    CPU FP64 reference: 누산/순서 차이를 최대한 줄인 reference.
    반환은 GPU fp32로 다시 올림.
    """
    device = x.device
    dtype = x.dtype
    B, D = x.shape

    x64 = x.detach().cpu().double()
    t64 = t.detach().cpu().double()
    W0_64 = W0.detach().cpu().double()
    b0_64 = b0.detach().cpu().double()
    W1_64 = W1.detach().cpu().double()
    b1_64 = b1.detach().cpu().double()

    lin0 = x64 @ W0_64.t() + b0_64
    relu0 = torch.relu(lin0)
    lin1 = relu0 @ W1_64.t() + b1_64

    dY = 2.0 * (lin1 - t64) / (B * D)

    d_relu0 = dY @ W1_64
    dW1 = dY.t() @ relu0
    db1 = dY.sum(dim=0)

    d_lin0 = d_relu0 * (relu0 > 0).double()
    dW0 = d_lin0.t() @ x64
    db0 = d_lin0.sum(dim=0)

    return (
        dW0.to(device=device, dtype=dtype),
        db0.to(device=device, dtype=dtype),
        dW1.to(device=device, dtype=dtype),
        db1.to(device=device, dtype=dtype),
    )


# -------------------------
# main
# -------------------------
def main():
    import aicf_fw.backend as BK
    from aicf_fw.backend.aicf_backend import AICFBackend

    BK.set_backend(AICFBackend())

    torch.manual_seed(0)

    # TF32 확실히 OFF (matmul 전에)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass

    print("[torch] allow_tf32.matmul =", torch.backends.cuda.matmul.allow_tf32)
    print("[torch] allow_tf32.cudnn  =", torch.backends.cudnn.allow_tf32)

    device = torch.device("cuda:0")
    dtype = torch.float32
    B, D = 64, 8

    # runtime tensors
    x = torch.randn(B, D, device=device, dtype=dtype)
    t = torch.randn(B, D, device=device, dtype=dtype)

    W0 = torch.randn(D, D, device=device, dtype=dtype)
    b0 = torch.randn(D, device=device, dtype=dtype)
    W1 = torch.randn(D, D, device=device, dtype=dtype)
    b1 = torch.randn(D, device=device, dtype=dtype)

    # optimizer state
    step0 = torch.zeros((), device=device, dtype=torch.int32)

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

        s_step = sym_tensor(name="opt.step", shape=tuple(step0.shape), dtype=step0.dtype, device=device)

        smW0 = sym_tensor(name="opt.m.0.W", shape=(D, D), dtype=dtype, device=device)
        svW0 = sym_tensor(name="opt.v.0.W", shape=(D, D), dtype=dtype, device=device)
        smb0 = sym_tensor(name="opt.m.0.b", shape=(D,), dtype=dtype, device=device)
        svb0 = sym_tensor(name="opt.v.0.b", shape=(D,), dtype=dtype, device=device)

        smW1 = sym_tensor(name="opt.m.2.W", shape=(D, D), dtype=dtype, device=device)
        svW1 = sym_tensor(name="opt.v.2.W", shape=(D, D), dtype=dtype, device=device)
        smb1 = sym_tensor(name="opt.m.2.b", shape=(D,), dtype=dtype, device=device)
        svb1 = sym_tensor(name="opt.v.2.b", shape=(D,), dtype=dtype, device=device)

        # fwd
        lin0 = linear(sx, sW0, sb0, name="lin0_out")
        r0 = relu(lin0, name="relu0_out")
        r0s = save(r0, name="relu0_saved")
        lin1 = linear(r0, sW1, sb1, name="lin1_out")
        dY = mse_grad(lin1, st, name="dY")

        # bwd
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

        # opt
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

    # compile / dump
    ir = trace_ir(build, name="v2_stage6_train1")
    print(dump_ir(ir))

    lowered = lower_to_backend_ops(ir)
    print(dump_lowered(lowered, title="LoweredOps(v2_stage6_train1)"))

    plan = build_binding_plan(ir)
    print(dump_plan(plan, title="BindingPlan(v2_stage6_train1)"))

    # run
    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        opts=ExecOptions(debug=False),
        device=device,
    )

    env = ex.run(
        inputs={"x": x, "t": t},
        params={
            "0.W": W0, "0.b": b0,
            "2.W": W1, "2.b": b1,
            "opt.step": step0,
            "opt.m.0.W": m_W0, "opt.v.0.W": v_W0,
            "opt.m.0.b": m_b0, "opt.v.0.b": v_b0,
            "opt.m.2.W": m_W1, "opt.v.2.W": v_W1,
            "opt.m.2.b": m_b1, "opt.v.2.b": v_b1,
        },
        reuse_static=True,
    )

    # invariants (shape/device strict, dtype warn-only, finite strict)
    check_env_invariants(ir, env, device=device)

    # coarse grad check (unit-op tests already passed; 여기서는 "대략"만)
    name_to_vid = {v.name: int(vid) for vid, v in ir.values.items()}

    dW0_ref, db0_ref, dW1_ref, db1_ref = torch_ref_grads_cpu_fp64(x, t, W0, b0, W1, b1)

    dW0_g = env[name_to_vid["d_0.W"]]
    db0_g = env[name_to_vid["d_0.b"]]
    dW1_g = env[name_to_vid["d_2.W"]]
    db1_g = env[name_to_vid["d_2.b"]]

    # 누산/순서 차이 감안한 넉넉한 tolerance
    assert_close("dW0", dW0_g, dW0_ref, atol=6e-2, rtol=6e-2)
    assert_close("db0", db0_g, db0_ref, atol=6e-2, rtol=6e-2)
    assert_close("dW1", dW1_g, dW1_ref, atol=3e-2, rtol=3e-2)
    assert_close("db1", db1_g, db1_ref, atol=2e-2, rtol=2e-2)

    print("[check] dW0 maxdiff =", mdiff(dW0_g, dW0_ref))
    print("[check] db0 maxdiff =", mdiff(db0_g, db0_ref))
    print("[check] dW1 maxdiff =", mdiff(dW1_g, dW1_ref))
    print("[check] db1 maxdiff =", mdiff(db1_g, db1_ref))

    print("OK (Stage6: invariants(shape/device) + grads coarse check)")


if __name__ == "__main__":
    main()
