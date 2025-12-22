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
def test_bias_add_f32_2d_lastdim(iters=200):
    """
    BiasAdd (v0.2-B safe generalized: last-dim only)

      inputs[0] = Y    f32 contiguous, rank>=2
      inputs[1] = bias f32 contiguous, rank=1, len == Y.shape[-1]
      outputs[0]= Out  same shape as Y, f32 contiguous, rank>=2

    Attr semantics:
      axis is optional, but if provided must be -1 or (rank-1).
      We pass axis=-1 explicitly for clarity.
    """
    M, N = 1024, 4096  # big enough so kernel shows clearly in ncu
    Y = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()
    Out = torch.empty_like(Y)

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.BiasAdd, [Y, bias], [Out], {"axis": -1})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::bias_add_f32_2d_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.BiasAdd, [Y, bias], [Out], {"axis": -1})
    torch.cuda.synchronize()

    ref = Y + bias
    check("BiasAdd(f32, 2D last-dim, axis=-1)", Out, ref)


@torch.inference_mode()
def test_bias_add_f32_4d_nhwc_lastdim(iters=200):
    """
    4D NHWC case (contiguous) should work with last-dim-only policy.

      Y: [B,H,W,C] contiguous
      bias: [C]
      Out: [B,H,W,C]
      axis=-1
    """
    B, H, W, C = 32, 8, 8, 256
    Y = torch.randn(B, H, W, C, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(C, device="cuda", dtype=torch.float32).contiguous()
    Out = torch.empty_like(Y)

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.BiasAdd, [Y, bias], [Out], {"axis": -1})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::bias_add_f32_4d_nhwc_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.BiasAdd, [Y, bias], [Out], {"axis": -1})
    torch.cuda.synchronize()

    ref = Y + bias
    check("BiasAdd(f32, NHWC last-dim, axis=-1)", Out, ref)


@torch.inference_mode()
def test_bias_add_f32_axis_mismatch_should_fail():
    """
    last-dim-only policy:
      - axis must be -1 or (rank-1)
    This should fail cleanly.
    """
    B, H, W, C = 8, 4, 4, 32
    Y = torch.randn(B, H, W, C, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(C, device="cuda", dtype=torch.float32).contiguous()
    Out = torch.empty_like(Y)

    try:
        op_call(aicf.OpKind.BiasAdd, [Y, bias], [Out], {"axis": 1})  # not last dim for NHWC
        torch.cuda.synchronize()
        raise RuntimeError("Expected failure for axis!=last-dim, but call succeeded.")
    except Exception as e:
        print(f"[BiasAdd axis!=last-dim] failed cleanly: {type(e).__name__}: {e}")


@torch.inference_mode()
def test_bias_add_cudagraph_capture_safe(iters=50, replays=200):
    """
    CUDA Graph capture/replay safety check using 2D input (simple and strict).
    """
    M, N = 256, 1024
    Y = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()
    Out = torch.empty_like(Y)

    # warmup allocator
    for _ in range(10):
        op_call(aicf.OpKind.BiasAdd, [Y, bias], [Out], {"axis": -1})
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()

    torch.cuda.synchronize()
    g.capture_begin()
    for _ in range(iters):
        op_call(aicf.OpKind.BiasAdd, [Y, bias], [Out], {"axis": -1})
    g.capture_end()
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::bias_add_cudagraph_replay"):
        for _ in range(replays):
            g.replay()
    torch.cuda.synchronize()

    ref = Y + bias
    check("BiasAdd(CUDAGraph)", Out, ref)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    # Focus targets for profiling / validation:
    test_bias_add_f32_2d_lastdim()
    test_bias_add_f32_4d_nhwc_lastdim()

    # Optional policy checks:
    # test_bias_add_f32_axis_mismatch_should_fail()

    # Optional capture-safe check:
    # test_bias_add_cudagraph_capture_safe()

    print("ALL OK")


# Example ncu:
# ncu --target-processes all --kernel-name "bias_add_f32_kernel" --launch-count 1 --section SpeedOfLight -o bias_add python .\test_add_bias_ncu.py
#
# If you want to profile the 4D case specifically, use --launch-count 1 and
# set the __main__ to call only test_bias_add_f32_4d_nhwc_lastdim().
#
# 메모리 바운드 커널 확인, 
# ncu --target-processes all --kernel-name "bias_add_f32_kernel" --launch-count 1     --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats     --page details -o bias_add_detail     python .\test_add_bias_ncu.py
#
# bias 가 캐시에 잘 올라가는지
# ncu --target-processes all --kernel-name "bias_add_f32_kernel" --launch-count 1     --section MemoryWorkloadAnalysis --section CacheHierarchy     --page details -o bias_add_cache     python .\test_add_bias_ncu.py
#
# 메모리 coalescing / transaction 확인
# ncu --target-processes all --kernel-name "bias_add_f32_kernel" --launch-count 1     --section MemoryWorkloadAnalysis --section MemoryAccessPattern     --page details -o bias_add_access     python .\test_add_bias_ncu.py
