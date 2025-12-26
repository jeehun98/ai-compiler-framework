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


def check(name, got_f16, ref_f32, atol=5e-2, rtol=5e-2):
    """
    got:  fp16 output from TC kernel (acc float -> store half)
    ref:  float32 reference
    compare in float32 space for stability
    """
    got = got_f16.float()
    ref = ref_f32.float()
    max_abs = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs} (got.dtype={got_f16.dtype})")
    if not ok:
        raise RuntimeError(f"{name} mismatch")


def op_call(kind, inputs, outputs, attrs=None):
    if attrs is None:
        attrs = {}
    aicf.op_call(kind, inputs, outputs, attrs)


@torch.inference_mode()
def test_gemm_f16_tc_out_f16_nn(iters=200):
    """
    GEMM NN: C[M,N] = A[M,K] * B[K,N]
    Contract (OUT_F16):
      - A: f16 contiguous [M,K]
      - B: f16 contiguous [K,N]
      - C: f16 contiguous [M,N]   <-- CHANGED
      - attrs: transA=False, transB=False
    """
    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_out_f16_nn_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check("Gemm(TC out_f16, NN)", C, ref)


@torch.inference_mode()
def test_gemm_f16_tc_out_f16_tn(iters=200):
    """
    GEMM TN: C[M,N] = (A_storage)^T * B
    Storage contract for transA=True (OUT_F16):
      - A_storage: f16 contiguous [K,M]  (already transposed storage)
      - B:        f16 contiguous [K,N]
      - C:        f16 contiguous [M,N]
      - attrs: transA=True, transB=False
    """
    M, N, K = 256, 256, 256

    A_logical = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    A_storage = A_logical.t().contiguous()  # [K,M]
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A_storage, B], [C], {"transA": True, "transB": False})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_out_f16_tn_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A_storage, B], [C], {"transA": True, "transB": False})
    torch.cuda.synchronize()

    # (A_storage)^T == A_logical
    ref = (A_logical.float() @ B.float()).contiguous()
    check("Gemm(TC out_f16, TN via transA=True storage)", C, ref)


@torch.inference_mode()
def test_gemm_f16_tc_out_f16_nt(iters=200):
    """
    GEMM NT: C[M,N] = A * (B_storage)^T
    Storage contract for transB=True (OUT_F16):
      - A:         f16 contiguous [M,K]
      - B_storage: f16 contiguous [N,K]  (already transposed storage)
      - C:         f16 contiguous [M,N]
      - attrs: transA=False, transB=True
    """
    M, N, K = 256, 256, 256

    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B_logical = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    B_storage = B_logical.t().contiguous()  # [N,K]
    C = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A, B_storage], [C], {"transA": False, "transB": True})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_out_f16_nt_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B_storage], [C], {"transA": False, "transB": True})
    torch.cuda.synchronize()

    # (B_storage)^T == B_logical
    ref = (A.float() @ B_logical.float()).contiguous()
    check("Gemm(TC out_f16, NT via transB=True storage)", C, ref)


@torch.inference_mode()
def test_gemm_f16_tc_out_f16_non_multiple_of_16(iters=50):
    """
    WMMA tile 16 기준이지만 OOB를 0 패딩하면 16 배수가 아닌 shape도 동작해야 함.
    out_f16는 마지막 float->half rounding이 들어가서 오차가 더 커질 수 있으니 atol/rtol을 더 넉넉히.
    """
    M, N, K = 123, 77, 65
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()

    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_out_f16_nonmul16_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B], [C], {"transA": False, "transB": False})
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check("Gemm(TC out_f16, NN non-multiple-of-16)", C, ref, atol=1e-1, rtol=1e-1)


@torch.inference_mode()
def test_gemm_f16_tc_out_f16_cudagraph_capture_safe(iters=10, replays=200):
    """
    CUDA Graph capture/replay safety check (out_f16).
    """
    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()

    # warmup allocator
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

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_out_f16_cudagraph_replay"):
        for _ in range(replays):
            g.replay()
    torch.cuda.synchronize()

    ref = (A.float() @ B.float()).contiguous()
    check("Gemm(TC out_f16, CUDAGraph)", C, ref)


if __name__ == "__main__":
    assert torch.cuda.is_available()
    torch.cuda.synchronize()

    # Core correctness tests
    test_gemm_f16_tc_out_f16_nn()
    test_gemm_f16_tc_out_f16_tn()
    test_gemm_f16_tc_out_f16_nt()

    # Optional robustness
    test_gemm_f16_tc_out_f16_non_multiple_of_16()

    # Optional capture-safe check
    # test_gemm_f16_tc_out_f16_cudagraph_capture_safe()

    print("ALL OK")
