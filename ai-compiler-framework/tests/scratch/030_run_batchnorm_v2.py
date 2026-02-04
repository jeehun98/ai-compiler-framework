from __future__ import annotations

import sys
from pathlib import Path
import torch

# ---- path bootstrap (aicf_v2 src) ----
THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]                 # python/aicf_v2
SRC = ROOT / "src"                     # python/aicf_v2/src
sp = str(SRC)
if sp not in sys.path:
    sys.path.insert(0, sp)

import aicf_v2 as aicf

# For NEG schema test (direct op_call)
from aicf_v2.backends.cuda.bridge import op_call, current_stream_u64

# Optional: print enums if _C is importable in your env
try:
    import _C
    HAS_C = True
except Exception:
    HAS_C = False


def max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run_fwd_training(N: int, C: int, H: int, W: int, *, affine: bool, eps: float = 1e-5) -> float:
    """
    v2 path:
      - BatchNormFwd(use_running_stats=False)
      - outputs: y, save_mean(fp32[C]), save_rstd(fp32[C])
    """
    m = aicf.Model(dtype="f16", device="cuda")
    spec_x = aicf.TensorSpec((N, C, H, W), "f16", "cuda")
    spec_c_f16 = aicf.TensorSpec((C,), "f16", "cuda")

    x = m.input("x", spec_x)

    if affine:
        gamma = m.input("gamma", spec_c_f16)
        beta  = m.input("beta",  spec_c_f16)
        y, save_mean, save_rstd = m.add(
            aicf.BatchNormFwd(name="bn", eps=eps, use_running_stats=False, affine=True),
            x, gamma, beta
        )
    else:
        y, save_mean, save_rstd = m.add(
            aicf.BatchNormFwd(name="bn", eps=eps, use_running_stats=False, affine=False),
            x
        )

    m.output("y", y)
    m.output("save_mean", save_mean)
    m.output("save_rstd", save_rstd)

    # feed
    xt = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    feed = {"x": xt}
    if affine:
        gt = torch.randn(C, device="cuda", dtype=torch.float16)
        bt = torch.randn(C, device="cuda", dtype=torch.float16)
        feed["gamma"] = gt
        feed["beta"] = bt

    exe = aicf.CudaExecutor()
    out = exe.run(m, feed)

    # torch ref (training=True, running stats None)
    if affine:
        y_ref = torch.nn.functional.batch_norm(
            xt, running_mean=None, running_var=None,
            weight=gt, bias=bt, training=True, momentum=0.0, eps=eps
        )
    else:
        y_ref = torch.nn.functional.batch_norm(
            xt, running_mean=None, running_var=None,
            weight=None, bias=None, training=True, momentum=0.0, eps=eps
        )

    d = max_abs(out["y"].float(), y_ref.float())
    return d


def run_fwd_infer(N: int, C: int, H: int, W: int, *, affine: bool, eps: float = 1e-5) -> float:
    """
    v2 path:
      - BatchNormFwd(use_running_stats=True)
      - outputs: y only
    """
    m = aicf.Model(dtype="f16", device="cuda")
    spec_x = aicf.TensorSpec((N, C, H, W), "f16", "cuda")
    spec_c_f16 = aicf.TensorSpec((C,), "f16", "cuda")
    spec_c_f32 = aicf.TensorSpec((C,), "f32", "cuda")

    x = m.input("x", spec_x)

    if affine:
        gamma = m.input("gamma", spec_c_f16)
        beta  = m.input("beta",  spec_c_f16)
        running_mean = m.input("running_mean", spec_c_f32)
        running_var  = m.input("running_var",  spec_c_f32)
        y = m.add(
            aicf.BatchNormFwd(name="bn", eps=eps, use_running_stats=True, affine=True),
            x, gamma, beta, running_mean, running_var
        )
    else:
        running_mean = m.input("running_mean", spec_c_f32)
        running_var  = m.input("running_var",  spec_c_f32)
        y = m.add(
            aicf.BatchNormFwd(name="bn", eps=eps, use_running_stats=True, affine=False),
            x, running_mean, running_var
        )

    m.output("y", y)

    # feed
    xt = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    rm = torch.randn(C, device="cuda", dtype=torch.float32)
    rv = torch.rand(C, device="cuda", dtype=torch.float32) + 0.5  # positive

    feed = {"x": xt, "running_mean": rm, "running_var": rv}

    if affine:
        gt = torch.randn(C, device="cuda", dtype=torch.float16)
        bt = torch.randn(C, device="cuda", dtype=torch.float16)
        feed["gamma"] = gt
        feed["beta"] = bt

    exe = aicf.CudaExecutor()
    out = exe.run(m, feed)

    # torch ref (training=False, use given running stats)
    y_ref = torch.nn.functional.batch_norm(
        xt, running_mean=rm, running_var=rv,
        weight=(gt if affine else None),
        bias=(bt if affine else None),
        training=False, momentum=0.0, eps=eps
    )

    d = max_abs(out["y"].float(), y_ref.float())
    return d


def run_bwd_training(N: int, C: int, H: int, W: int, *, eps: float = 1e-5) -> float:
    """
    v2 path in a single graph:
      - fwd(training, affine) produces save_mean/save_rstd
      - dy is input
      - bwd consumes x, dy, gamma, save_mean, save_rstd
      - outputs: dx(f16), dgamma(f32), dbeta(f32)
    """
    m = aicf.Model(dtype="f16", device="cuda")

    spec_x = aicf.TensorSpec((N, C, H, W), "f16", "cuda")
    spec_c_f16 = aicf.TensorSpec((C,), "f16", "cuda")

    x = m.input("x", spec_x)
    dy = m.input("dy", spec_x)

    gamma = m.input("gamma", spec_c_f16)
    beta  = m.input("beta",  spec_c_f16)

    # forward (training affine) -> y, save_mean, save_rstd
    y, save_mean, save_rstd = m.add(
        aicf.BatchNormFwd(name="bn", eps=eps, use_running_stats=False, affine=True),
        x, gamma, beta
    )
    # backward -> dx, dgamma, dbeta
    dx, dgamma, dbeta = m.add(
        aicf.BatchNormBwd(name="bn_bwd"),
        x, dy, gamma, save_mean, save_rstd
    )

    m.output("dx", dx)
    m.output("dgamma", dgamma)
    m.output("dbeta", dbeta)

    # feed
    with torch.no_grad():
        xt = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
        gt = torch.randn(C, device="cuda", dtype=torch.float16)
        bt = torch.randn(C, device="cuda", dtype=torch.float16)
        dyt = torch.randn_like(xt)

    exe = aicf.CudaExecutor()
    out = exe.run(m, {"x": xt, "dy": dyt, "gamma": gt, "beta": bt})

    # torch ref gradients (must be autograd-enabled)
    x2 = xt.detach().clone().requires_grad_(True)
    g2 = gt.detach().clone().requires_grad_(True)
    b2 = bt.detach().clone().requires_grad_(True)

    y_ref = torch.nn.functional.batch_norm(
        x2, running_mean=None, running_var=None,
        weight=g2, bias=b2,
        training=True, momentum=0.0, eps=eps
    )
    y_ref.backward(dyt)  # dy doesn't need grad

    d_dx = max_abs(out["dx"].float(), x2.grad.float())
    d_dg = max_abs(out["dgamma"], g2.grad.float())
    d_db = max_abs(out["dbeta"],  b2.grad.float())
    return float(max(d_dx, d_dg, d_db))


def neg_wrong_rank() -> None:
    """
    v2 NEG: wrong rank should fail in layer checks (or kernel).
    """
    try:
        m = aicf.Model(dtype="f16", device="cuda")
        x = m.input("x", aicf.TensorSpec((8, 16), "f16", "cuda"))  # wrong rank
        y = m.add(aicf.BatchNormFwd(name="bn", eps=1e-5, use_running_stats=True, affine=False), x)  # should raise
        m.output("y", y)
        exe = aicf.CudaExecutor()
        xt = torch.randn(8, 16, device="cuda", dtype=torch.float16)
        exe.run(m, {"x": xt})
        print("[NEG rank] unexpected success")
    except Exception as e:
        print("[NEG rank] ok:", str(e).splitlines()[0])


def neg_wrong_schema() -> None:
    """
    Direct call NEG: wrong schema id should be rejected by kernel.
    (v2 graph path always uses registry-provided schema, so we force it here.)
    """
    if not HAS_C:
        print("[NEG schema] skipped: _C not importable in this env")
        return

    try:
        # Minimal infer noaff call contract:
        # inputs: [x, running_mean(fp32), running_var(fp32)] or [x]?? (depends on kernel)
        # We'll use infer noaff: [x, rm, rv] -> [y]
        N, C, H, W = 8, 16, 8, 8
        x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16).contiguous()
        y = torch.empty_like(x).contiguous()
        rm = torch.zeros(C, device="cuda", dtype=torch.float32).contiguous()
        rv = torch.ones(C, device="cuda", dtype=torch.float32).contiguous()

        wrong_schema = 0x12345678
        wrong_blob = b"\x00" * 8

        # kind_id for BatchNormFwd is 15 (per your enum)
        # stream: current
        op_call(
            kind_id=int(_C.OpKind.BatchNormFwd),
            inputs=[x, rm, rv],
            outputs=[y],
            attr_schema=wrong_schema,
            attr_blob=wrong_blob,
            stream=current_stream_u64(),
        )
        print("[NEG schema] unexpected success")
    except Exception as e:
        print("[NEG schema] ok:", str(e).splitlines()[0])


def main():
    if HAS_C:
        print(f"BatchNormFwd enum value = {int(_C.OpKind.BatchNormFwd)}")
        print(f"BatchNormBwd enum value = {int(_C.OpKind.BatchNormBwd)}")
    else:
        print("(_C not importable here; running v2 tests only)")

    worst = 0.0

    shapes = [(8, 16, 8, 8), (16, 32, 16, 16), (7, 33, 5, 7)]
    for (N, C, H, W) in shapes:
        d = run_fwd_training(N, C, H, W, affine=True, eps=1e-5)
        print(f"[FWD train affine] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

        d = run_fwd_training(N, C, H, W, affine=False, eps=1e-5)
        print(f"[FWD train noaff ] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

        d = run_fwd_infer(N, C, H, W, affine=True, eps=1e-5)
        print(f"[FWD infer affine] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

        d = run_fwd_infer(N, C, H, W, affine=False, eps=1e-5)
        print(f"[FWD infer noaff ] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

        d = run_bwd_training(N, C, H, W, eps=1e-5)
        print(f"[BWD train affine] N={N} C={C} H={H} W={W} max|d|={d:.3e}")
        worst = max(worst, d)

    print(f"[OK] worst max|delta| = {worst}")

    # NEG tests
    neg_wrong_rank()
    neg_wrong_schema()


if __name__ == "__main__":
    main()
