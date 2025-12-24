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


def check(name, got, ref, atol=1e-3, rtol=1e-3):
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
    """
    SGDStep f32: W -= lr * dW
    - Warmup / profiling loop: iters updates (for ncu visibility)
    - Correctness: reset and check SINGLE step exactly
    """
    numel = 1 << 20
    lr = 1e-3

    W0 = torch.randn(numel, device="cuda", dtype=torch.float32).contiguous()
    dW = torch.randn_like(W0).contiguous()

    # --- warmup allocator + dispatch path ---
    W = W0.clone()
    for _ in range(10):
        op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
    torch.cuda.synchronize()

    # --- profiling loop (not used for correctness) ---
    with torch.cuda.nvtx.range("AICF::sgd_step_f32_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
    torch.cuda.synchronize()

    # --- correctness: single step ---
    W1 = W0.clone()
    op_call(aicf.OpKind.SgdStep, [W1, dW], [W1], {"lr": float(lr)})
    torch.cuda.synchronize()

    ref = (W0 - lr * dW).contiguous()
    check("SgdStep(f32, 1-step)", W1, ref)


@torch.inference_mode()
def test_sgd_step_f16_scalar_forced(iters=200):
    """
    Force scalar f16 variant by making numel odd (half2 requires even numel).
    - Warmup/profiling: iters updates
    - Correctness: reset and check SINGLE step
    """
    numel = (1 << 20) + 1  # odd
    lr = 1e-3

    W0 = torch.randn(numel, device="cuda", dtype=torch.float16).contiguous()
    dW = torch.randn_like(W0).contiguous()

    # warmup
    W = W0.clone()
    for _ in range(10):
        op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
    torch.cuda.synchronize()

    # profiling loop
    with torch.cuda.nvtx.range("AICF::sgd_step_f16_scalar_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
    torch.cuda.synchronize()

    # correctness: single step
    W1 = W0.clone()
    op_call(aicf.OpKind.SgdStep, [W1, dW], [W1], {"lr": float(lr)})
    torch.cuda.synchronize()

    ref = (W0.float() - lr * dW.float()).half().contiguous()
    check("SgdStep(f16 scalar forced, 1-step)", W1, ref, atol=5e-3, rtol=5e-3)


@torch.inference_mode()
def test_sgd_step_f16_half2_preferred(iters=200):
    """
    f16 half2 should be preferred when:
      - numel even
      - pointers are 4B aligned (PyTorch contiguous half tensor usually is)
    - Warmup/profiling: iters updates
    - Correctness: reset and check SINGLE step
    """
    numel = 1 << 20  # even
    lr = 1e-3

    W0 = torch.randn(numel, device="cuda", dtype=torch.float16).contiguous()
    dW = torch.randn_like(W0).contiguous()

    # warmup
    W = W0.clone()
    for _ in range(10):
        op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
    torch.cuda.synchronize()

    # profiling loop
    with torch.cuda.nvtx.range("AICF::sgd_step_f16_half2_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
    torch.cuda.synchronize()

    # correctness: single step
    W1 = W0.clone()
    op_call(aicf.OpKind.SgdStep, [W1, dW], [W1], {"lr": float(lr)})
    torch.cuda.synchronize()

    ref = (W0.float() - lr * dW.float()).half().contiguous()
    check("SgdStep(f16 half2 preferred, 1-step)", W1, ref, atol=5e-3, rtol=5e-3)


@torch.inference_mode()
def test_sgd_step_cudagraph_capture_safe(iters=10, replays=200):
    """
    Capture/replay safety check (use f16 even numel so half2 likely triggers).
    Here correctness must match iters updates performed during capture.
    """
    numel = 1 << 18
    lr = 1e-3

    W0 = torch.randn(numel, device="cuda", dtype=torch.float16).contiguous()
    dW = torch.randn_like(W0).contiguous()
    W  = W0.clone()

    # allocator warmup
    for _ in range(10):
        op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()

    torch.cuda.synchronize()
    g.capture_begin()
    for _ in range(iters):
        op_call(aicf.OpKind.SgdStep, [W, dW], [W], {"lr": float(lr)})
    g.capture_end()
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::sgd_step_cudagraph_replay"):
        for _ in range(replays):
            g.replay()
    torch.cuda.synchronize()

    # reference: capture ran iters updates once; replay runs iters updates per replay
    total_steps = iters * (1 + replays)
    ref = W0.float()
    for _ in range(total_steps):
        ref = ref - lr * dW.float()
    ref = ref.half().contiguous()

    check("SgdStep(CUDAGraph)", W, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    test_sgd_step_f32()
    test_sgd_step_f16_scalar_forced()
    test_sgd_step_f16_half2_preferred()
    # Optional:
    # test_sgd_step_cudagraph_capture_safe()

    print("ALL OK")


# Example ncu:
# - Profile ONE path at a time by commenting out other tests in __main__.
#
# 1) f32
# ncu --target-processes all --kernel-name "sgd_step_f32_kernel" --launch-count 1 --section SpeedOfLight -o sgd_f32 python .\test_sgd_step_ncu.py
#
# 2) f16 scalar (odd numel)
# ncu --target-processes all --kernel-name "sgd_step_f16_kernel" --launch-count 1 --section SpeedOfLight -o sgd_f16_scalar python .\test_sgd_step_ncu.py
#
# 3) f16 half2 (even numel)
# ncu --target-processes all --kernel-name "sgd_step_f16_half2_kernel" --launch-count 1 --section SpeedOfLight -o sgd_f16_half2 python .\test_sgd_step_ncu.py
