import os
import sys
from pathlib import Path

# NOTE:
# This test lives under examples/python/python_binding_test/
# so repo root is 3 parents up from this file.
REPO_ROOT = Path(__file__).resolve().parents[3]
PYMOD_DIR  = REPO_ROOT / "build" / "python"
PKG_DIR    = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))

if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import torch
import aicf_cuda as aicf


def check(name, got, ref, atol=1e-4, rtol=1e-4):
    max_abs = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs}")
    if not ok:
        raise RuntimeError(f"{name} mismatch")


def op_call(kind, inputs, outputs, attrs=None):
    if attrs is None:
        attrs = {}
    aicf.op_call(kind, inputs, outputs, attrs)


@torch.inference_mode()
def train_step_linear_bias_mse_sgd(M=256, K=512, N=128, lr=1e-2, iters=10):
    """
    Minimal 1-train-step cycle (f32-only path) using existing ops:

      Y  = X @ W              (Gemm f32 naive)
      Z  = Y + b              (BiasAdd)
      dZ = (2/numel)*(Z - T)  (MseGrad default scale)
      db = sum(dZ, axis=0)    (ReduceSum last-dim)
      dW = X^T @ dZ           (Gemm f32 naive, using materialized Xt)
      W -= lr * dW            (SgdStep f32)
      b -= lr * db            (SgdStep f32)

    Notes:
    - We materialize transposes with .transpose(...).contiguous() because
      current GEMM variants don't support stride-view transpose safely.
    - This is for "works end-to-end" verification, not for TC performance.
    """

    # ---- tensors (f32 only) ----
    X = torch.randn(M, K, device="cuda", dtype=torch.float32).contiguous()
    W = torch.randn(K, N, device="cuda", dtype=torch.float32).contiguous()
    b = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()
    T = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()

    # ---- buffers ----
    Y  = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()
    Z  = torch.empty_like(Y)
    dZ = torch.empty_like(Y)

    dW = torch.empty(K, N, device="cuda", dtype=torch.float32).contiguous()
    db = torch.empty(N, device="cuda", dtype=torch.float32).contiguous()

    # ---- optional: baseline loss for sanity ----
    def mse_mean(pred, target):
        return ((pred - target) ** 2).mean()

    # warmup allocator + kernels
    for _ in range(5):
        op_call(aicf.OpKind.Gemm, [X, W], [Y], {"transB": False})
        op_call(aicf.OpKind.BiasAdd, [Y, b], [Z], {"axis": -1})
        op_call(aicf.OpKind.MseGrad, [Z, T], [dZ], {})
        op_call(aicf.OpKind.ReduceSum, [dZ], [db], {"axis": -1})
        Xt = X.transpose(0, 1).contiguous()
        op_call(aicf.OpKind.Gemm, [Xt, dZ], [dW], {"transB": False})
        op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
        op_call(aicf.OpKind.SgdStep, [b, db], [b], {"lr": float(lr)})
    torch.cuda.synchronize()

    # measure loss before
    op_call(aicf.OpKind.Gemm, [X, W], [Y], {"transB": False})
    op_call(aicf.OpKind.BiasAdd, [Y, b], [Z], {"axis": -1})
    torch.cuda.synchronize()
    loss0 = mse_mean(Z, T).item()
    print(f"[loss] before = {loss0:.6f}")

    # ---- training loop ----
    with torch.cuda.nvtx.range("AICF::train_step_1cycle_f32_loop"):
        for _ in range(iters):
            # forward
            op_call(aicf.OpKind.Gemm, [X, W], [Y], {"transB": False})
            op_call(aicf.OpKind.BiasAdd, [Y, b], [Z], {"axis": -1})

            # dZ
            op_call(aicf.OpKind.MseGrad, [Z, T], [dZ], {})  # default scale=2/numel

            # db
            op_call(aicf.OpKind.ReduceSum, [dZ], [db], {"axis": -1})

            # dW = X^T @ dZ  (materialize Xt)
            Xt = X.transpose(0, 1).contiguous()
            op_call(aicf.OpKind.Gemm, [Xt, dZ], [dW], {"transB": False})

            # update
            op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
            op_call(aicf.OpKind.SgdStep, [b, db], [b], {"lr": float(lr)})

    torch.cuda.synchronize()

    # measure loss after
    op_call(aicf.OpKind.Gemm, [X, W], [Y], {"transB": False})
    op_call(aicf.OpKind.BiasAdd, [Y, b], [Z], {"axis": -1})
    torch.cuda.synchronize()
    loss1 = mse_mean(Z, T).item()
    print(f"[loss] after  = {loss1:.6f}")

    print("DONE train steps:", iters)
    print("LOSS delta:", loss1 - loss0)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    train_step_linear_bias_mse_sgd()

    print("ALL OK")


# Optional ncu example (profile one GEMM launch):
# ncu --target-processes all --kernel-name "gemm_f32_naive_kernel" --launch-count 1 --section SpeedOfLight -o train_step_gemm python .\test_train_step_1cycle.py
