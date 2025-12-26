from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# This test lives under examples/python/python_binding_test/
REPO_ROOT = Path(__file__).resolve().parents[3]
PYMOD_DIR  = REPO_ROOT / "build" / "python"
PKG_DIR    = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))

if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import torch
import aicf_cuda as aicf


# -------------------------
# helpers
# -------------------------
def op_call(kind, inputs, outputs, attrs: Optional[Dict[str, Any]] = None):
    aicf.op_call(kind, inputs, outputs, attrs or {})


def expect(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def assert_contig(t: torch.Tensor, name: str) -> None:
    expect(t.is_contiguous(), f"{name} must be contiguous (got strides={t.stride()})")


def assert_shape(t: torch.Tensor, shape: Tuple[int, ...], name: str) -> None:
    expect(tuple(t.shape) == tuple(shape), f"{name} shape mismatch: got {tuple(t.shape)} expected {tuple(shape)}")


def assert_dtype(t: torch.Tensor, dtype: torch.dtype, name: str) -> None:
    expect(t.dtype == dtype, f"{name} dtype mismatch: got {t.dtype} expected {dtype}")


def check_close(name: str, got: torch.Tensor, ref: torch.Tensor, atol=1e-2, rtol=1e-2) -> None:
    assert_contig(got, "got")
    assert_contig(ref, "ref")
    expect(got.shape == ref.shape, f"{name}: shape mismatch got={tuple(got.shape)} ref={tuple(ref.shape)}")
    expect(got.dtype == ref.dtype, f"{name}: dtype mismatch got={got.dtype} ref={ref.dtype}")

    diff = (got - ref).abs()
    max_abs = diff.max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs} shape={tuple(got.shape)} dtype={got.dtype}")
    if not ok:
        # print a small diagnostic
        idx = diff.argmax().item()
        flat_g = got.view(-1)
        flat_r = ref.view(-1)
        print(f"  worst idx={idx} got={flat_g[idx].item()} ref={flat_r[idx].item()} abs={max_abs}")
        raise RuntimeError(f"{name} mismatch")


def warmup_gemm(A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, attrs: Dict[str, Any], iters: int = 10):
    for _ in range(iters):
        op_call(aicf.OpKind.Gemm, [A, B], [C], attrs)
    torch.cuda.synchronize()


# -------------------------
# tests
# -------------------------
@torch.inference_mode()
def test_gemm_tc_nn(M=256, N=256, K=256, iters: int = 200):
    """
    NN contract (TC path):
      A: f16 [M,K] contiguous
      B: f16 [K,N] contiguous
      C: f32 [M,N] contiguous
      attrs: transA=False, transB=False
    """
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    assert_contig(A, "A"); assert_contig(B, "B"); assert_contig(C, "C")
    assert_shape(A, (M, K), "A"); assert_shape(B, (K, N), "B"); assert_shape(C, (M, N), "C")
    assert_dtype(A, torch.float16, "A"); assert_dtype(B, torch.float16, "B"); assert_dtype(C, torch.float32, "C")

    attrs = {"transA": False, "transB": False}
    warmup_gemm(A, B, C, attrs)

    with torch.cuda.nvtx.range("AICF::GemmTC::NN"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B], [C], attrs)
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check_close("Gemm(TC f16->f32) NN", C, ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_gemm_tc_tn_storage(M=256, N=256, K=256, iters: int = 200):
    """
    TN storage contract:
      We want: C = (A_storage)^T @ B
      If A_storage = A_logical.t() and A_logical is [M,K],
      then (A_storage)^T == A_logical
      so ref: A_logical @ B

    Contract:
      A_storage: f16 [K,M] contiguous
      B:         f16 [K,N] contiguous
      C:         f32 [M,N] contiguous
      attrs: transA=True, transB=False
    """
    A_logical = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    A_storage = A_logical.t().contiguous()  # [K,M]
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    assert_shape(A_storage, (K, M), "A_storage"); assert_shape(B, (K, N), "B"); assert_shape(C, (M, N), "C")
    assert_dtype(A_storage, torch.float16, "A_storage"); assert_dtype(B, torch.float16, "B"); assert_dtype(C, torch.float32, "C")

    attrs = {"transA": True, "transB": False}
    warmup_gemm(A_storage, B, C, attrs)

    with torch.cuda.nvtx.range("AICF::GemmTC::TN_storage"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A_storage, B], [C], attrs)
    torch.cuda.synchronize()

    ref = (A_logical.float() @ B.float()).contiguous()
    check_close("Gemm(TC f16->f32) TN (transA=True storage[K,M])", C, ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_gemm_tc_nt_storage(M=256, N=256, K=256, iters: int = 200):
    """
    NT storage contract:
      We want: C = A @ (B_storage)^T
      If B_storage = B_logical.t() and B_logical is [K,N],
      then (B_storage)^T == B_logical
      so ref: A @ B_logical

    Contract:
      A:         f16 [M,K] contiguous
      B_storage: f16 [N,K] contiguous
      C:         f32 [M,N] contiguous
      attrs: transA=False, transB=True
    """
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B_logical = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    B_storage = B_logical.t().contiguous()  # [N,K]
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    assert_shape(A, (M, K), "A"); assert_shape(B_storage, (N, K), "B_storage"); assert_shape(C, (M, N), "C")
    assert_dtype(A, torch.float16, "A"); assert_dtype(B_storage, torch.float16, "B_storage"); assert_dtype(C, torch.float32, "C")

    attrs = {"transA": False, "transB": True}
    warmup_gemm(A, B_storage, C, attrs)

    with torch.cuda.nvtx.range("AICF::GemmTC::NT_storage"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B_storage], [C], attrs)
    torch.cuda.synchronize()

    ref = (A.float() @ B_logical.float()).contiguous()
    check_close("Gemm(TC f16->f32) NT (transB=True storage[N,K])", C, ref, atol=1e-2, rtol=1e-2)


@torch.inference_mode()
def test_gemm_tc_nonmul16(M=123, N=77, K=65, iters: int = 50):
    """
    Non-multiple-of-16: correctness with OOB zero-padding.
    Expect slightly looser tolerance in general, but your current max_abs was very small.
    """
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    attrs = {"transA": False, "transB": False}
    warmup_gemm(A, B, C, attrs)

    with torch.cuda.nvtx.range("AICF::GemmTC::NonMul16"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B], [C], attrs)
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check_close("Gemm(TC f16->f32) NN nonmul16", C, ref, atol=5e-2, rtol=5e-2)


@torch.inference_mode()
def test_gemm_tc_cudagraph_capture_safe(M=256, N=256, K=256, iters: int = 10, replays: int = 200):
    """
    CUDA Graph capture/replay safety check (Torch CUDAGraph).
    """
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    attrs = {"transA": False, "transB": False}
    warmup_gemm(A, B, C, attrs)

    g = torch.cuda.CUDAGraph()
    torch.cuda.synchronize()

    g.capture_begin()
    for _ in range(iters):
        op_call(aicf.OpKind.Gemm, [A, B], [C], attrs)
    g.capture_end()
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::GemmTC::CUDAGraphReplay"):
        for _ in range(replays):
            g.replay()
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check_close("Gemm(TC f16->f32) CUDAGraph", C, ref, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    # Core correctness tests
    test_gemm_tc_nn()
    test_gemm_tc_tn_storage()
    test_gemm_tc_nt_storage()

    # Robustness
    test_gemm_tc_nonmul16()

    # Optional capture-safe check
    # test_gemm_tc_cudagraph_capture_safe()

    print("ALL OK")
