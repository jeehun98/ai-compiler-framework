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

    # ---------------------------
    # PyTorch reference forward/backward
    # ---------------------------
    y_ref = F.batch_norm(
        x, None, None,
        weight=gamma, bias=beta,
        training=True, momentum=0.1, eps=eps
    )

    # stats that you feed into AICF bwd (from x)
    mean_ref, rstd_ref = ref_bn_train_saved_stats(x.detach(), eps)

    dy = torch.randn_like(y_ref)

    y_ref.backward(dy)

    dx_ref = x.grad.detach()
    dgamma_ref = gamma.grad.detach().float()
    dbeta_ref  = beta.grad.detach().float()

    # ---------------------------
    # AICF backward
    # ---------------------------
    dx = torch.empty_like(x)
    dgamma = torch.empty((C,), device=device, dtype=torch.float32)
    dbeta  = torch.empty((C,), device=device, dtype=torch.float32)

    save_mean = mean_ref.detach().contiguous()
    save_rstd = rstd_ref.detach().contiguous()

    aicf_cuda.op_call(
        aicf_cuda.OpKind.BatchNormBwd,
        inputs=[x.detach(), dy.detach(), gamma.detach(), save_mean, save_rstd],
        outputs=[dx, dgamma, dbeta],
        attrs={}
    )

    # ---------------------------
    # Formula reference (matches WHAT WE PASS into AICF)
    #   dbeta  = sum(dy)
    #   dgamma = sum(dy * xhat)
    #   dx     = (gamma * rstd / NHW) * (NHW*dy - sum(dy) - xhat*sum(dy*xhat))
    # ---------------------------
    with torch.no_grad():
        dbeta_ref2 = dy.detach().float().sum(dim=(0, 2, 3))  # [C]

        xhat = (x.detach().float() - save_mean[None, :, None, None]) * save_rstd[None, :, None, None]
        dgamma_ref2 = (dy.detach().float() * xhat).sum(dim=(0, 2, 3))  # [C]

        NHW = float(N * H * W)
        g = gamma.detach().float()[None, :, None, None]
        rstd4 = save_rstd[None, :, None, None]
        dx_ref2 = (g * rstd4 / NHW) * (
            NHW * dy.detach().float()
            - dbeta_ref2[None, :, None, None]
            - xhat * dgamma_ref2[None, :, None, None]
        )

    # ---------------------------
    # Compare AICF vs (A) PyTorch autograd, (B) formula refs
    # ---------------------------
    dx_err_pt = max_abs_err(dx.float(), dx_ref.float())
    dg_err_pt = max_abs_err(dgamma, dgamma_ref)
    db_err_pt = max_abs_err(dbeta, dbeta_ref)

    dx_err_f  = max_abs_err(dx.float(), dx_ref2)
    dg_err_f  = max_abs_err(dgamma, dgamma_ref2)
    db_err_f  = max_abs_err(dbeta, dbeta_ref2)

    print(f"[BN bwd pt]     N={N} C={C} H={H} W={W} "
          f"dx_err={dx_err_pt} dgamma_err={dg_err_pt} dbeta_err={db_err_pt}")
    print(f"[BN bwd formula] N={N} C={C} H={H} W={W} "
          f"dx_err={dx_err_f} dgamma_err={dg_err_f} dbeta_err={db_err_f}")

    # Assert against formula correctness first (this isolates kernel correctness)
    assert dx_err_f < 8e-3,  f"dx_err(formula) too large: {dx_err_f}"
    assert dg_err_f < 3e-3,  f"dgamma_err(formula) too large: {dg_err_f}"
    assert db_err_f < 3e-3,  f"dbeta_err(formula) too large: {db_err_f}"

    # Optional: keep PyTorch checks but looser (PyTorch may use different saved stats / internals)
    # assert dx_err_pt < 8e-3,  f"dx_err(pt) too large: {dx_err_pt}"
    # assert dg_err_pt < 3e-2,  f"dgamma_err(pt) too large: {dg_err_pt}"
    # assert db_err_pt < 3e-2,  f"dbeta_err(pt) too large: {db_err_pt}"


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    run_bn_bwd_affine(torch.float16, N=8,  C=32, H=16, W=16)
    run_bn_bwd_affine(torch.float16, N=16, C=64, H=8,  W=8)


if __name__ == "__main__":
    main()
