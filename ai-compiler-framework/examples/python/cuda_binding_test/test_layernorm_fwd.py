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


import torch
import torch.nn.functional as F
import aicf_cuda


def _max_abs(a, b):
    return (a - b).abs().max().item()


def ln_call(x, gamma=None, beta=None, eps=1e-5):
    """
    Call AICF LayerNorm forward via op_call.
    inputs: [x] or [x, gamma, beta]
    outputs: [y, mean, rstd]
    """
    assert x.is_cuda and x.is_contiguous() and x.dim() == 2
    M, N = x.shape

    # outputs
    y = torch.empty_like(x)
    mean = torch.empty((M,), device=x.device, dtype=torch.float32)
    rstd = torch.empty((M,), device=x.device, dtype=torch.float32)

    inputs = [x] if gamma is None or beta is None else [x, gamma, beta]
    outputs = [y, mean, rstd]
    attrs = {"eps": float(eps)}

    aicf_cuda._C.op_call(aicf_cuda._C.OpKind.LayerNorm, inputs, outputs, attrs)
    return y, mean, rstd


def test_layernorm_fwd_correctness(dtype=torch.float16, device="cuda"):
    torch.manual_seed(0)

    shapes = [(128, 256), (64, 1024), (32, 4096)]
    eps = 1e-5

    for (M, N) in shapes:
        x = torch.randn((M, N), device=device, dtype=dtype).contiguous()
        gamma = torch.ones((N,), device=device, dtype=dtype).contiguous()
        beta  = torch.zeros((N,), device=device, dtype=dtype).contiguous()

        # affine on
        y_a, mean_a, rstd_a = ln_call(x, gamma, beta, eps)
        y_t = F.layer_norm(x, (N,), gamma, beta, eps=eps)

        tol = 5e-3 if dtype == torch.float16 else 1e-5
        err = _max_abs(y_a.float(), y_t.float())
        print(f"[affine] M={M} N={N} dtype={dtype} max_abs_err={err}")
        assert err < tol

        # affine off
        y_a2, mean_a2, rstd_a2 = ln_call(x, None, None, eps)
        y_t2 = F.layer_norm(x, (N,), None, None, eps=eps)

        err2 = _max_abs(y_a2.float(), y_t2.float())
        print(f"[noaff] M={M} N={N} dtype={dtype} max_abs_err={err2}")
        assert err2 < tol

        # mean/rstd sanity (optional, light)
        mu = x.float().mean(dim=1)
        var = x.float().var(dim=1, unbiased=False)
        rstd_ref = torch.rsqrt(var + eps)

        mu_err = _max_abs(mean_a, mu)
        rs_err = _max_abs(rstd_a, rstd_ref)
        print(f"[stats] mean_err={mu_err} rstd_err={rs_err}")
        assert mu_err < (3e-3 if dtype == torch.float16 else 1e-5)
        assert rs_err < (3e-3 if dtype == torch.float16 else 1e-5)


def test_layernorm_fwd_cudagraph(dtype=torch.float16, device="cuda"):
    """
    Uses AICF binding's capture_begin/end/replay.
    Key rule: tensor pointers must be stable across replay.
    So we allocate x/gamma/beta/y/mean/rstd once and reuse them.
    """
    torch.manual_seed(1)

    M, N = 128, 1024
    eps = 1e-5

    x = torch.randn((M, N), device=device, dtype=dtype).contiguous()
    gamma = torch.randn((N,), device=device, dtype=dtype).contiguous()
    beta  = torch.randn((N,), device=device, dtype=dtype).contiguous()

    y = torch.empty_like(x)
    mean = torch.empty((M,), device=device, dtype=torch.float32)
    rstd = torch.empty((M,), device=device, dtype=torch.float32)

    inputs = [x, gamma, beta]
    outputs = [y, mean, rstd]
    attrs = {"eps": float(eps)}

    # warmup (optional)
    aicf_cuda._C.op_call(aicf_cuda._C.OpKind.LayerNorm, inputs, outputs, attrs)
    torch.cuda.synchronize()

    # capture
    aicf_cuda._C.capture_begin()
    aicf_cuda._C.op_call(aicf_cuda._C.OpKind.LayerNorm, inputs, outputs, attrs)
    aicf_cuda._C.capture_end()

    # replay a few times
    for i in range(20):
        aicf_cuda._C.replay()
    torch.cuda.synchronize()

    # validate against torch once (same x/gamma/beta)
    y_t = F.layer_norm(x, (N,), gamma, beta, eps=eps)
    tol = 5e-3 if dtype == torch.float16 else 1e-5
    err = _max_abs(y.float(), y_t.float())
    print(f"[cudagraph] max_abs_err={err}")
    assert err < tol

    # cleanup
    aicf_cuda._C.capture_reset()


if __name__ == "__main__":
    test_layernorm_fwd_correctness(torch.float16)
    test_layernorm_fwd_cudagraph(torch.float16)
    print("OK")
