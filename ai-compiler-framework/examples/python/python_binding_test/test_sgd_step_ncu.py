import os, sys
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


def check(name, got, ref, atol=1e-5, rtol=1e-5):
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
def test_sgd_step_f32(iters=200):
    M, N = 1024, 4096
    param = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    grad  = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()

    lr = 1e-2
    ref = param - lr * grad

    # in-place update: output points to same tensor
    for _ in range(10):
        op_call(aicf.OpKind.SgdStep, [param, grad], [param], {"lr": float(lr)})
    torch.cuda.synchronize()

    # reset
    param2 = (ref + lr * grad).contiguous()  # reconstruct original param
    ref2 = param2 - lr * grad

    with torch.cuda.nvtx.range("AICF::sgd_step_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.SgdStep, [param2, grad], [param2], {"lr": float(lr)})
    torch.cuda.synchronize()

    # after iters updates: param2 = orig - iters*lr*grad
    ref_iters = (ref2 - (iters - 1) * lr * grad)  # since ref2 already did 1 step
    check("SgdStep(f32, in-place)", param2, ref_iters)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()
    test_sgd_step_f32()
    print("ALL OK")


# ncu:
# ncu --target-processes all --kernel-name "sgd_step_f32_kernel" --launch-count 1 --section SpeedOfLight -o sgd_step python .\test_sgd_step_ncu.py
