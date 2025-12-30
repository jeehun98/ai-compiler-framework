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
def ref_bn_train_saved_stats(x, eps):
    # per-channel over N*H*W, unbiased=False
    dims = (0, 2, 3)
    mean = x.float().mean(dim=dims)                 # [C]
    var  = x.float().var(dim=dims, unbiased=False)  # [C]
    rstd = torch.rsqrt(var + eps)                   # [C]
    return mean, rstd


def run_bn_bwd_affine(dtype=torch.float16, N=8, C=16, H=7, W=7, eps=1e-5):
    device = "cuda"

    x = (torch.randn((N, C, H, W), device=device, dtype=dtype) * 0.5).requires_grad_(True)
    gamma = (torch.randn((C,), device=device, dtype=dtype) * 0.1 + 1.0).requires_grad_(True)
    beta  = (torch.randn((C,), device=device, dtype=dtype) * 0.1).requires_grad_(True)

    # forward (training)
    y_ref = F.batch_norm(
        x, None, None,
        weight=gamma, bias=beta,
        training=True, momentum=0.1, eps=eps
    )

    # saved stats: match your kernel definition (mean/rstd from x)
    mean_ref, rstd_ref = ref_bn_train_saved_stats(x.detach(), eps)

    # random upstream grad
    dy = torch.randn_like(y_ref)

    # PyTorch grads: inject upstream grad directly
    y_ref.backward(dy)

    dx_ref = x.grad.detach()
    dgamma_ref = gamma.grad.detach().float()
    dbeta_ref  = beta.grad.detach().float()

    # ---- AICF run ----
    dx = torch.empty_like(x)
    dgamma = torch.empty((C,), device=device, dtype=torch.float32)
    dbeta  = torch.empty((C,), device=device, dtype=torch.float32)

    save_mean = mean_ref.detach().contiguous()
    save_rstd = rstd_ref.detach().contiguous()

    # binding in your snippet: op_call(kind, inputs, outputs, attrs=dict)
    aicf_cuda.op_call(
        aicf_cuda.OpKind.BatchNormBwd,
        inputs=[x.detach(), dy.detach(), gamma.detach(), save_mean, save_rstd],
        outputs=[dx, dgamma, dbeta],
        attrs={}
    )

    # errors
    dx_err = max_abs_err(dx.float(), dx_ref.float())
    dg_err = max_abs_err(dgamma, dgamma_ref)
    db_err = max_abs_err(dbeta, dbeta_ref)

    print(f"[BN bwd f16 affine] N={N} C={C} H={H} W={W} "
          f"dx_err={dx_err} dgamma_err={dg_err} dbeta_err={db_err}")

    # tolerances: atomic accumulation + fp16 I/O
    assert dx_err < 8e-3, f"dx_err too large: {dx_err}"
    assert dg_err < 3e-2, f"dgamma_err too large: {dg_err}"
    assert db_err < 3e-2, f"dbeta_err too large: {db_err}"


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    run_bn_bwd_affine(torch.float16, N=8,  C=32, H=16, W=16)
    run_bn_bwd_affine(torch.float16, N=16, C=64, H=8,  W=8)


if __name__ == "__main__":
    main()
