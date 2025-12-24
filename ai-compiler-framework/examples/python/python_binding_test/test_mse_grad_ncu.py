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
def test_mse_grad_f32_default_scale(iters=200):
    M, N = 1024, 4096
    pred = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    target = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    dPred = torch.empty_like(pred)

    for _ in range(10):
        op_call(aicf.OpKind.MseGrad, [pred, target], [dPred], {})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::mse_grad_default_scale_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.MseGrad, [pred, target], [dPred], {})
    torch.cuda.synchronize()

    ref = (pred - target) * (2.0 / pred.numel())
    check("MseGrad(f32, default scale=2/numel)", dPred, ref)


@torch.inference_mode()
def test_mse_grad_f32_custom_scale(iters=200):
    M, N = 512, 2048
    pred = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    target = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    dPred = torch.empty_like(pred)

    scale = 0.1234

    for _ in range(10):
        op_call(aicf.OpKind.MseGrad, [pred, target], [dPred], {"scale": float(scale)})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::mse_grad_custom_scale_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.MseGrad, [pred, target], [dPred], {"scale": float(scale)})
    torch.cuda.synchronize()

    ref = (pred - target) * scale
    check("MseGrad(f32, custom scale)", dPred, ref)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    test_mse_grad_f32_default_scale()
    test_mse_grad_f32_custom_scale()

    print("ALL OK")


# ncu:
# ncu --target-processes all --kernel-name "mse_grad_f32_kernel" --launch-count 1 --section SpeedOfLight -o mse_grad python .\test_mse_grad_ncu.py
