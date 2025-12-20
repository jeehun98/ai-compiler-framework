import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
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
def test_add_f32(iters=200):
    N = 1024
    a = torch.randn(N, device="cuda", dtype=torch.float32)
    b = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty_like(a)

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.EltwiseAdd, [a, b], [out], {})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::add_f32_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.EltwiseAdd, [a, b], [out], {})
    torch.cuda.synchronize()

    ref = a + b
    check("EltwiseAdd(f32)", out, ref)


@torch.inference_mode()
def test_add_f16_vec2_expected(iters=400):
    # even N -> should hit vec2 half2 variant if registered
    N = 4096  # bigger so kernel shows clearly in ncu
    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty_like(a)

    for _ in range(10):
        op_call(aicf.OpKind.EltwiseAdd, [a, b], [out], {})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::add_f16_even_should_vec2"):
        for _ in range(iters):
            op_call(aicf.OpKind.EltwiseAdd, [a, b], [out], {})
    torch.cuda.synchronize()

    ref = a + b
    check("EltwiseAdd(f16 even)", out, ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_add_f16_naive_fallback(iters=400):
    # odd N -> vec2 variant should be unsupported -> fallback to naive f16
    N = 4097
    a = torch.randn(N, device="cuda", dtype=torch.float16)
    b = torch.randn(N, device="cuda", dtype=torch.float16)
    out = torch.empty_like(a)

    for _ in range(10):
        op_call(aicf.OpKind.EltwiseAdd, [a, b], [out], {})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::add_f16_odd_should_fallback_naive"):
        for _ in range(iters):
            op_call(aicf.OpKind.EltwiseAdd, [a, b], [out], {})
    torch.cuda.synchronize()

    ref = a + b
    check("EltwiseAdd(f16 odd)", out, ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_relu_f32(iters=200):
    N = 4096
    x = torch.randn(N, device="cuda", dtype=torch.float32)
    out = torch.empty_like(x)

    for _ in range(10):
        op_call(aicf.OpKind.EltwiseRelu, [x], [out], {})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::relu_f32_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.EltwiseRelu, [x], [out], {})
    torch.cuda.synchronize()

    ref = torch.relu(x)
    check("EltwiseRelu(f32)", out, ref)


@torch.inference_mode()
def test_gemm_f32(iters=40):
    M, K, N = 128, 256, 96
    A = torch.randn(M, K, device="cuda", dtype=torch.float32)
    B = torch.randn(K, N, device="cuda", dtype=torch.float32)
    C = torch.empty(M, N, device="cuda", dtype=torch.float32)

    for _ in range(3):
        op_call(aicf.OpKind.Gemm, [A, B], [C], {})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f32_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B], [C], {})
    torch.cuda.synchronize()

    ref = A @ B
    check("Gemm(f32)", C, ref, atol=2e-3, rtol=2e-3)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    # pick what you want to profile (keep it focused)
    test_add_f16_vec2_expected()
    test_add_f16_naive_fallback()

    # optional:
    # test_add_f32()
    # test_relu_f32()
    # test_gemm_f32()

    print("ALL OK")


# ncu --target-processes all --kernel-name "add_f16x2_kernel" --launch-count 1 --section SpeedOfLight -o add_vec2 python .\test_aicf_variants_ncu.py


