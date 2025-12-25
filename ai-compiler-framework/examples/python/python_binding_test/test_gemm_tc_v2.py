# examples/python/python_binding_test/test_gemm_tc_v2.py
from __future__ import annotations

import os
import sys
from pathlib import Path

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


def check(name: str, got: torch.Tensor, ref: torch.Tensor, atol=1e-2, rtol=1e-2) -> None:
    max_abs = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs} got={tuple(got.shape)}/{got.dtype} ref={tuple(ref.shape)}/{ref.dtype}")
    if not ok:
        raise RuntimeError(f"{name} mismatch (max_abs={max_abs})")


def op_call(kind, inputs, outputs, attrs=None):
    if attrs is None:
        attrs = {}
    aicf.op_call(kind, inputs, outputs, attrs)


@torch.inference_mode()
def test_gemm_f16_tc_nn(iters: int = 200):
    """
    NN: C[M,N] = A[M,K] * B[K,N]
    inputs: A f16 [M,K] contig, B f16 [K,N] contig
    output: C f32 [M,N] contig
    attrs: transA=False, transB=False
    """
    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_nn_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check("Gemm(f16 TC, NN)", C, ref)


@torch.inference_mode()
def test_gemm_f16_tc_tn(iters: int = 200):
    """
    TN: C[M,N] = (A_storage)^T * B
    storage contract when transA=True:
      A_storage: f16 [K,M] contiguous  (already transposed storage)
      B:         f16 [K,N] contiguous
      C:         f32 [M,N] contiguous
    attrs: transA=True, transB=False

    If A_storage == A_logical.t(), then (A_storage)^T == A_logical
    so ref is: A_logical @ B
    """
    M, N, K = 256, 256, 256
    A_logical = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    A_storage = A_logical.t().contiguous()  # [K,M]
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A_storage, B], [C], {"transA": True, "transB": False})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_tn_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A_storage, B], [C], {"transA": True, "transB": False})
    torch.cuda.synchronize()

    ref = (A_logical.float() @ B.float()).contiguous()
    check("Gemm(f16 TC, TN via transA=True storage)", C, ref)


@torch.inference_mode()
def test_gemm_f16_tc_nt(iters: int = 200):
    """
    NT: C[M,N] = A * (B_storage)^T
    storage contract when transB=True:
      A:         f16 [M,K] contiguous
      B_storage: f16 [N,K] contiguous (already transposed storage)
      C:         f32 [M,N] contiguous
    attrs: transA=False, transB=True

    If B_storage == B_logical.t(), then (B_storage)^T == B_logical
    so ref is: A @ B_logical
    """
    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B_logical = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    B_storage = B_logical.t().contiguous()  # [N,K]
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A, B_storage], [C], {"transA": False, "transB": True})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_nt_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B_storage], [C], {"transA": False, "transB": True})
    torch.cuda.synchronize()

    ref = (A.float() @ B_logical.float()).contiguous()
    check("Gemm(f16 TC, NT via transB=True storage)", C, ref)


@torch.inference_mode()
def test_gemm_f16_tc_non_multiple_of_16(iters: int = 50):
    """
    Non-multiple-of-16 shapes should work if kernel zero-pads OOB.
    Your launcher uses ceil16(K) loop + OOB checks => should pass.
    """
    M, N, K = 123, 77, 65
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_nonmul16_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check("Gemm(f16 TC, NN non-multiple-of-16)", C, ref, atol=5e-2, rtol=5e-2)


@torch.inference_mode()
def test_gemm_f16_tc_cudagraph_capture_safe(iters: int = 10, replays: int = 200):
    """
    CUDA Graph capture/replay safety check.
    (Note: Your WMMA kernel uses __shared__ arrays; that's fine for capture.
           The key is: no dynamic allocations inside op_call path.)
    """
    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    g = torch.cuda.CUDAGraph()

    torch.cuda.synchronize()
    g.capture_begin()
    for _ in range(iters):
        op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    g.capture_end()
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_cudagraph_replay"):
        for _ in range(replays):
            g.replay()
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check("Gemm(f16 TC, CUDAGraph)", C, ref)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    # Core correctness tests
    test_gemm_f16_tc_nn()
    test_gemm_f16_tc_tn()
    test_gemm_f16_tc_nt()

    # Optional robustness
    test_gemm_f16_tc_non_multiple_of_16()

    # Optional capture-safe check
    # test_gemm_f16_tc_cudagraph_capture_safe()

    print("ALL OK")
