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
def test_relu_bwd_f32(iters=200):
    M, N = 1024, 4096
    Y = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    dOut = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    dY = torch.empty_like(Y)

    for _ in range(10):
        op_call(aicf.OpKind.ReluBwd, [Y, dOut], [dY], {})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::relu_bwd_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.ReluBwd, [Y, dOut], [dY], {})
    torch.cuda.synchronize()

    ref = dOut * (Y > 0).to(torch.float32)
    check("ReluBwd(f32)", dY, ref)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()
    test_relu_bwd_f32()
    print("ALL OK")


# ncu:
# ncu --target-processes all --kernel-name "relu_bwd_f32_kernel" --launch-count 1 --section SpeedOfLight -o relu_bwd python .\test_relu_bwd_ncu.py
