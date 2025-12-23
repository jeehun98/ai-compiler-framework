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
def test_reduce_sum_f32_2d_lastdim(iters=200):
    """
    ReduceSum (last-dim-only policy)

      inputs[0]  = dY  f32 contiguous, rank>=2
      outputs[0] = dB  f32 contiguous, rank=1, len == dY.shape[-1]

    Attr semantics:
      axis is optional, but if provided must be -1 or (rank-1).
      We pass axis=-1 explicitly for clarity.
    """
    M, N = 4096, 2048  # big enough for profiling
    dY = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    dB = torch.empty(N, device="cuda", dtype=torch.float32).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.ReduceSum, [dY], [dB], {"axis": -1})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::reduce_sum_f32_2d_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.ReduceSum, [dY], [dB], {"axis": -1})
    torch.cuda.synchronize()

    ref = dY.sum(dim=0)
    check("ReduceSum(f32, 2D last-dim, axis=-1)", dB, ref)


@torch.inference_mode()
def test_reduce_sum_f32_4d_nhwc_lastdim(iters=200):
    """
    4D NHWC case (contiguous) should work with last-dim-only policy.

      dY: [B,H,W,C] contiguous
      dB: [C]
      axis=-1
    """
    B, H, W, C = 32, 8, 8, 256
    dY = torch.randn(B, H, W, C, device="cuda", dtype=torch.float32).contiguous()
    dB = torch.empty(C, device="cuda", dtype=torch.float32).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.ReduceSum, [dY], [dB], {"axis": -1})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::reduce_sum_f32_4d_nhwc_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.ReduceSum, [dY], [dB], {"axis": -1})
    torch.cuda.synchronize()

    ref = dY.sum(dim=(0, 1, 2))  # sum over everything except last dim
    check("ReduceSum(f32, NHWC last-dim, axis=-1)", dB, ref)


@torch.inference_mode()
def test_reduce_sum_axis_mismatch_should_fail():
    """
    last-dim-only policy:
      - axis must be -1 or (rank-1)
    This should fail cleanly.
    """
    B, H, W, C = 8, 4, 4, 32
    dY = torch.randn(B, H, W, C, device="cuda", dtype=torch.float32).contiguous()
    dB = torch.empty(C, device="cuda", dtype=torch.float32).contiguous()

    try:
        op_call(aicf.OpKind.ReduceSum, [dY], [dB], {"axis": 1})  # not last dim for NHWC
        torch.cuda.synchronize()
        raise RuntimeError("Expected failure for axis!=last-dim, but call succeeded.")
    except Exception as e:
        print(f"[ReduceSum axis!=last-dim] failed cleanly: {type(e).__name__}: {e}")


@torch.inference_mode()
def test_reduce_sum_cudagraph_capture_safe(iters=50, replays=200):
    """
    CUDA Graph capture/replay safety check.
    """
    M, N = 1024, 1024
    dY = torch.randn(M, N, device="cuda", dtype=torch.float32).contiguous()
    dB = torch.empty(N, device="cuda", dtype=torch.float32).contiguous()

    # warmup allocator
    for _ in range(10):
        op_call(aicf.OpKind.ReduceSum, [dY], [dB], {"axis": -1})
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()

    torch.cuda.synchronize()
    g.capture_begin()
    for _ in range(iters):
        op_call(aicf.OpKind.ReduceSum, [dY], [dB], {"axis": -1})
    g.capture_end()
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::reduce_sum_cudagraph_replay"):
        for _ in range(replays):
            g.replay()
    torch.cuda.synchronize()

    ref = dY.sum(dim=0)
    check("ReduceSum(CUDAGraph)", dB, ref)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    # Focus targets for profiling / validation:
    test_reduce_sum_f32_2d_lastdim()
    test_reduce_sum_f32_4d_nhwc_lastdim()

    # Optional policy checks:
    # test_reduce_sum_axis_mismatch_should_fail()

    # Optional capture-safe check:
    # test_reduce_sum_cudagraph_capture_safe()

    print("ALL OK")


# Example ncu:
# ncu --target-processes all --kernel-name "reduce_sum_lastdim_f32_kernel" --launch-count 1 --section SpeedOfLight -o reduce_sum python .\test_reduce_sum_ncu.py
#
# Details (memory bound / access pattern):
# ncu --target-processes all --kernel-name "reduce_sum_lastdim_f32_kernel" --launch-count 1 ^
#     --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats --page details ^
#     -o reduce_sum_detail python .\test_reduce_sum_ncu.py
#
# Cache check:
# ncu --target-processes all --kernel-name "reduce_sum_lastdim_f32_kernel" --launch-count 1 ^
#     --section MemoryWorkloadAnalysis --section CacheHierarchy --page details ^
#     -o reduce_sum_cache python .\test_reduce_sum_ncu.py
#
# Memory coalescing / transactions:
# ncu --target-processes all --kernel-name "reduce_sum_lastdim_f32_kernel" --launch-count 1 ^
#     --section MemoryWorkloadAnalysis --section MemoryAccessPattern --page details ^
#     -o reduce_sum_access python .\test_reduce_sum_ncu.py
