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


def op_call(kind, inputs, outputs, attrs=None):
    if attrs is None:
        attrs = {}
    aicf.op_call(kind, inputs, outputs, attrs)


def check_all_zero(name, t: torch.Tensor):
    mx = t.abs().max().item() if t.numel() else 0.0
    ok = (mx == 0.0)
    print(f"[{name}] ok={ok} max_abs={mx}")
    if not ok:
        raise RuntimeError(f"{name} expected all zeros, but max_abs={mx}")


@torch.inference_mode()
def test_grad_zero_f32_1d(iters=200):
    n = 1_000_000
    g = torch.randn(n, device="cuda", dtype=torch.float32).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.GradZero, [g], [g])
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::grad_zero_f32_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.GradZero, [g], [g])
    torch.cuda.synchronize()

    check_all_zero("GradZero(f32, 1D)", g)


@torch.inference_mode()
def test_grad_zero_f16_2d(iters=200):
    M, N = 4096, 4096
    g = torch.randn(M, N, device="cuda", dtype=torch.float16).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.GradZero, [g], [g])
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::grad_zero_f16_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.GradZero, [g], [g])
    torch.cuda.synchronize()

    check_all_zero("GradZero(f16, 2D)", g)


@torch.inference_mode()
def test_grad_zero_noncontig_should_fail():
    """
    MVP 정책: contiguous만 지원.
    non-contiguous가 들어오면 supported()에서 false → dispatch 실패로 예외가 나야 정상.
    """
    g0 = torch.randn(1024, 1024, device="cuda", dtype=torch.float32).contiguous()
    g = g0.t()  # non-contiguous view

    try:
        op_call(aicf.OpKind.GradZero, [g], [g])
        torch.cuda.synchronize()
        raise RuntimeError("Expected failure for non-contiguous, but call succeeded.")
    except Exception as e:
        print(f"[GradZero non-contig] failed cleanly: {type(e).__name__}: {e}")


@torch.inference_mode()
def test_grad_zero_cudagraph_capture_safe(iters=50, replays=200, dtype=torch.float32):
    n = 1_000_000
    g = torch.randn(n, device="cuda", dtype=dtype).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.GradZero, [g], [g])
    torch.cuda.synchronize()

    # ✅ non-default stream 준비
    s = torch.cuda.Stream()
    cg = torch.cuda.CUDAGraph()

    # ✅ 캡처는 반드시 이 스트림에서
    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        cg.capture_begin()
        for _ in range(iters):
            op_call(aicf.OpKind.GradZero, [g], [g])
        cg.capture_end()
    torch.cuda.synchronize()

    # 값 다시 채우고 replay로 0 되는지 확인
    g.copy_(torch.randn_like(g))
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::grad_zero_cudagraph_replay"):
        for _ in range(replays):
            cg.replay()
    torch.cuda.synchronize()

    check_all_zero(f"GradZero(CUDAGraph, {str(dtype)})", g)


@torch.inference_mode()
def test_grad_zero_aicf_capture_safe(iters=50, replays=200, dtype=torch.float32):
    """
    너희 aicf_cuda._C의 capture_begin/end/replay로 capture/replay 가능한지 확인.
    (dedicated stream 정책 포함)
    """
    n = 1_000_000
    g = torch.randn(n, device="cuda", dtype=dtype).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.GradZero, [g], [g])
    torch.cuda.synchronize()

    aicf.capture_begin()
    for _ in range(iters):
        op_call(aicf.OpKind.GradZero, [g], [g])
    aicf.capture_end()
    torch.cuda.synchronize()

    # 값 다시 채워서 replay로 0 되는지 확인
    g.copy_(torch.randn_like(g))
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::grad_zero_aicf_replay"):
        for _ in range(replays):
            aicf.replay()
    torch.cuda.synchronize()

    check_all_zero(f"GradZero(AICF Graph, {str(dtype)})", g)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    # 기본 correctness
    test_grad_zero_f32_1d()
    test_grad_zero_f16_2d()

    # 정책 테스트 (옵션)
    test_grad_zero_noncontig_should_fail()

    # capture-safe 테스트 (옵션)
    test_grad_zero_cudagraph_capture_safe(dtype=torch.float32)
    test_grad_zero_cudagraph_capture_safe(dtype=torch.float16)

    # AICF graph 캡처/리플레이 테스트 (옵션)
    test_grad_zero_aicf_capture_safe(dtype=torch.float32)
    test_grad_zero_aicf_capture_safe(dtype=torch.float16)

    print("ALL OK")

# Example ncu:
# ncu --target-processes all --launch-count 1 --section SpeedOfLight \
#   --page details -o grad_zero_detail python .\test_grad_zero.py
#
# Memset은 커널명이 아니라 driver API 노드로 잡힐 수 있어서,
# --kernel-name 필터 대신 launch-count로 보는 게 안정적임.
