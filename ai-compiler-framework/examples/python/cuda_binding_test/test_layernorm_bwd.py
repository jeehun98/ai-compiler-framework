
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PYMOD_DIR  = REPO_ROOT / "build" / "python"
PKG_DIR    = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))

if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

# ======================================================================
# examples/python/cuda_binding_test/test_layernorm_bwd.py
# - Matches AICF launcher contract exactly:
#   affine=True : inputs=[x, dy, gamma, mean, rstd], outputs=[dx, dgamma(f32), dbeta(f32)]
#   affine=False: inputs=[x, dy, mean, rstd],       outputs=[dx]
# - Reference uses torch.nn.functional.layer_norm (one backward per graph)
# - Prints max_abs_err for dx / dgamma / dbeta and asserts tolerances
# ======================================================================

import torch
import torch.nn.functional as F

import aicf_cuda
from aicf_cuda import OpKind


def _max_abs(a, b):
    return (a - b).abs().max().item()


def _run_ref(x, gamma, beta, dy, eps, affine: bool):
    # Make independent graph for reference
    x2 = x.detach().clone().requires_grad_(True)
    g2 = None
    b2 = None
    if affine:
        g2 = gamma.detach().clone().requires_grad_(True)
        b2 = beta.detach().clone().requires_grad_(True)

    y2 = F.layer_norm(x2, (x2.shape[1],), weight=g2, bias=b2, eps=eps)
    # single backward
    y2.backward(dy.to(dtype=y2.dtype))

    dx_ref = x2.grad.detach()
    if affine:
        dgamma_ref = g2.grad.detach().to(torch.float32)
        dbeta_ref = b2.grad.detach().to(torch.float32)
        return dx_ref, dgamma_ref, dbeta_ref
    return dx_ref, None, None


def _run_aicf(x, gamma, beta, dy, eps, affine: bool):
    # Forward: produce mean/rstd needed by backward
    M, N = x.shape
    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    attrs = {"eps": float(eps)}

    if affine:
        aicf_cuda.op_call(OpKind.LayerNormFwd, [x, gamma, beta], [y, mean, rstd], attrs)
    else:
        aicf_cuda.op_call(OpKind.LayerNormFwd, [x], [y, mean, rstd], attrs)

    # Backward
    dx = torch.empty_like(x)

    if affine:
        dgamma = torch.empty((N,), device=x.device, dtype=torch.float32)
        dbeta = torch.empty((N,), device=x.device, dtype=torch.float32)

        aicf_cuda.op_call(
            OpKind.LayerNormBwd,
            [x, dy, gamma, mean, rstd],
            [dx, dgamma, dbeta],
            {},
        )
        return dx, dgamma, dbeta

    aicf_cuda.op_call(
        OpKind.LayerNormBwd,
        [x, dy, mean, rstd],
        [dx],
        {},
    )
    return dx, None, None


def test_layernorm_bwd(dtype=torch.float16, affine=True, M=128, N=256, eps=1e-5):
    assert dtype in (torch.float16, torch.float32)

    device = "cuda"
    torch.manual_seed(0)

    # Inputs (contiguous contract)
    x = (torch.randn((M, N), device=device, dtype=dtype) * 0.5).contiguous()
    dy = (torch.randn((M, N), device=device, dtype=dtype) * 0.5).contiguous()

    if affine:
        gamma = (torch.randn((N,), device=device, dtype=dtype) * 0.5).contiguous()
        beta = (torch.randn((N,), device=device, dtype=dtype) * 0.5).contiguous()
    else:
        gamma = None
        beta = None

    # AICF
    if affine:
        dx, dgamma, dbeta = _run_aicf(x, gamma, beta, dy, eps, affine=True)
    else:
        dx, dgamma, dbeta = _run_aicf(x, None, None, dy, eps, affine=False)

    # REF
    dx_ref, dgamma_ref, dbeta_ref = _run_ref(x, gamma, beta, dy, eps, affine=affine)

    # Compare
    dx_err = _max_abs(dx.float(), dx_ref.float())

    print(f"[dx] dtype={dtype} affine={affine} M={M} N={N} max_abs_err={dx_err}")

    # Tolerances (you can tighten later)
    if dtype == torch.float32:
        tol_dx = 5e-5
        tol_gb = 5e-5
    else:
        # fp16 accum errors are larger; dx is stored fp16 in AICF
        tol_dx = 2e-2 if N >= 4096 else 5e-3
        tol_gb = 1e-2   

    assert dx_err < tol_dx, f"dx_err too large: {dx_err} (tol {tol_dx})"

    if affine:
        dg_err = _max_abs(dgamma, dgamma_ref)
        db_err = _max_abs(dbeta, dbeta_ref)
        print(f"[dgamma] max_abs_err={dg_err}")
        print(f"[dbeta ] max_abs_err={db_err}")

        assert dg_err < tol_gb, f"dgamma_err too large: {dg_err} (tol {tol_gb})"
        assert db_err < tol_gb, f"dbeta_err too large: {db_err} (tol {tol_gb})"


def main():
    # Small-ish
    test_layernorm_bwd(torch.float16, affine=True,  M=128, N=256)
    test_layernorm_bwd(torch.float16, affine=False, M=128, N=256)

    # Medium
    test_layernorm_bwd(torch.float16, affine=True,  M=64,  N=1024)
    test_layernorm_bwd(torch.float16, affine=False, M=64,  N=1024)

    # Large N
    test_layernorm_bwd(torch.float16, affine=True,  M=32,  N=4096)
    test_layernorm_bwd(torch.float16, affine=False, M=32,  N=4096)

    # Optional f32
    test_layernorm_bwd(torch.float32, affine=True,  M=64, N=256)
    test_layernorm_bwd(torch.float32, affine=False, M=64, N=256)

    print("OK")


if __name__ == "__main__":
    main()
