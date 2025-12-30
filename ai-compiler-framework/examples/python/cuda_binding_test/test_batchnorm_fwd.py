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


def max_abs_err(a, b):
    return (a - b).abs().max().item()


@torch.no_grad()
def ref_bn_fwd_infer(x, gamma, beta, running_mean, running_var, eps):
    y = F.batch_norm(
        x, running_mean, running_var,
        weight=gamma, bias=beta,
        training=False, momentum=0.1, eps=eps
    )
    return y


def ref_bn_fwd_train(x, gamma, beta, eps):
    y = F.batch_norm(
        x, None, None,
        weight=gamma, bias=beta,
        training=True, momentum=0.1, eps=eps
    )
    dims = (0, 2, 3)
    mean = x.float().mean(dim=dims)  # [C]
    var = x.float().var(dim=dims, unbiased=False)  # [C]
    rstd = torch.rsqrt(var + eps)
    return y, mean, rstd


def run_bn_fwd_infer(dtype=torch.float16, affine=True, N=8, C=16, H=7, W=7, eps=1e-5):
    device = "cuda"
    x = torch.randn((N, C, H, W), device=device, dtype=dtype) * 0.5

    if affine:
        gamma = torch.randn((C,), device=device, dtype=dtype) * 0.1 + 1.0
        beta  = torch.randn((C,), device=device, dtype=dtype) * 0.1
    else:
        gamma = None
        beta = None

    running_mean = torch.randn((C,), device=device, dtype=torch.float32) * 0.1
    running_var  = torch.rand((C,), device=device, dtype=torch.float32) * 0.5 + 0.5

    y_ref = ref_bn_fwd_infer(x, gamma, beta, running_mean, running_var, eps)

    # ---- AICF run ----
    y = torch.empty_like(x)

    attrs = {
        "eps": float(eps),
        "use_running_stats": True,
    }

    if affine:
        aicf_cuda.op_call(
            aicf_cuda.OpKind.BatchNormFwd,
            inputs=[x, gamma, beta, running_mean, running_var],
            outputs=[y],
            attrs=attrs
        )
    else:
        aicf_cuda.op_call(
            aicf_cuda.OpKind.BatchNormFwd,
            inputs=[x, running_mean, running_var],
            outputs=[y],
            attrs=attrs
        )

    err = max_abs_err(y.float(), y_ref.float())
    print(f"[BN fwd infer] dtype={dtype} affine={affine} N={N} C={C} H={H} W={W} max_abs_err={err}")
    assert err < 3e-3, f"y_err too large: {err}"


def run_bn_fwd_train(dtype=torch.float16, affine=True, N=8, C=16, H=7, W=7, eps=1e-5):
    device = "cuda"
    x = torch.randn((N, C, H, W), device=device, dtype=dtype) * 0.5

    if affine:
        gamma = torch.randn((C,), device=device, dtype=dtype) * 0.1 + 1.0
        beta  = torch.randn((C,), device=device, dtype=dtype) * 0.1
    else:
        gamma = None
        beta = None

    y_ref, mean_ref, rstd_ref = ref_bn_fwd_train(x, gamma, beta, eps)

    # ---- AICF run ----
    y = torch.empty_like(x)
    save_mean = torch.empty((C,), device=device, dtype=torch.float32)
    save_rstd = torch.empty((C,), device=device, dtype=torch.float32)

    attrs = {
        "eps": float(eps),
        "use_running_stats": False,
    }

    if affine:
        aicf_cuda.op_call(
            aicf_cuda.OpKind.BatchNormFwd,
            inputs=[x, gamma, beta],
            outputs=[y, save_mean, save_rstd],
            attrs=attrs
        )
    else:
        aicf_cuda.op_call(
            aicf_cuda.OpKind.BatchNormFwd,
            inputs=[x],
            outputs=[y, save_mean, save_rstd],
            attrs=attrs
        )

    y_err = max_abs_err(y.float(), y_ref.float())
    m_err = max_abs_err(save_mean, mean_ref)
    r_err = max_abs_err(save_rstd, rstd_ref)

    print(f"[BN fwd train] dtype={dtype} affine={affine} N={N} C={C} H={H} W={W} "
          f"y_err={y_err} mean_err={m_err} rstd_err={r_err}")

    assert y_err < 3e-3, f"y_err too large: {y_err}"
    assert m_err < 5e-4, f"mean_err too large: {m_err}"
    assert r_err < 5e-4, f"rstd_err too large: {r_err}"


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    run_bn_fwd_infer(torch.float16, affine=True,  N=8, C=32, H=16, W=16)
    run_bn_fwd_infer(torch.float16, affine=False, N=8, C=32, H=16, W=16)

    run_bn_fwd_train(torch.float16, affine=True,  N=8, C=32, H=16, W=16)
    run_bn_fwd_train(torch.float16, affine=False, N=8, C=32, H=16, W=16)


if __name__ == "__main__":
    main()
