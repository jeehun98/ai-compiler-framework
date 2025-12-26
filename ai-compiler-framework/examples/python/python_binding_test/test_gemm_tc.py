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


def check(name, got, ref, atol=1e-2, rtol=1e-2):
    # fp16 TC path + float accum이라도 수치 오차가 있을 수 있어 atol/rtol 조금 넉넉히
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
def test_gemm_f16_tc_nn(iters=200):
    """
    GEMM NN: C[M,N] = A[M,K] * B[K,N]
    Contract (TC variant baseline):
      - A: f16 contiguous [M,K]
      - B: f16 contiguous [K,N]
      - C: f32 contiguous [M,N]
      - attrs: transA=False, transB=False
    """
    M, N, K = 256, 256, 256  # 16의 배수로 시작 (WMMA bring-up 최적)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    # warmup
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
def test_gemm_f16_tc_tn(iters=200):
    """
    GEMM TN: C[M,N] = A^T[M,K] * B[K,N] where A^T is logical transpose of A_storage.
    Storage-shape contract for transA=True baseline:
      - A_storage: f16 contiguous [K,M]  (already transposed storage)
      - B:        f16 contiguous [K,N]
      - C:        f32 contiguous [M,N]
      - attrs: transA=True, transB=False
    """
    M, N, K = 256, 256, 256
    # Create logical A as [M,K], then store A_storage as [K,M] contiguous
    A_logical = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    A_storage = A_logical.t().contiguous()  # [K,M] contiguous
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A_storage, B], [C], {"transA": True, "transB": False})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_tn_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A_storage, B], [C], {"transA": True, "transB": False})
    torch.cuda.synchronize()

    ref = (A_logical.float().t() @ B.float()).t().contiguous()
    # 설명:
    # 우리가 원하는 건 C = (A_storage)^T * B
    # A_storage는 A_logical.t() 이므로 (A_storage)^T == A_logical
    # 즉 C = A_logical * B
    ref = (A_logical.float() @ B.float()).contiguous()

    check("Gemm(f16 TC, TN via transA=True storage)", C, ref)


@torch.inference_mode()
def test_gemm_f16_tc_nt(iters=200):
    """
    GEMM NT: C[M,N] = A[M,K] * B^T[K,N] where B^T is logical transpose of B_storage.
    Storage-shape contract for transB=True baseline:
      - A:        f16 contiguous [M,K]
      - B_storage:f16 contiguous [N,K]  (already transposed storage)
      - C:        f32 contiguous [M,N]
      - attrs: transA=False, transB=True
    """
    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    # Create logical B as [K,N], then store B_storage as [N,K] contiguous
    B_logical = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    B_storage = B_logical.t().contiguous()  # [N,K] contiguous
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.Gemm, [A, B_storage], [C], {"transA": False, "transB": True})
    torch.cuda.synchronize()

    with torch.cuda.nvtx.range("AICF::gemm_f16_tc_nt_loop"):
        for _ in range(iters):
            op_call(aicf.OpKind.Gemm, [A, B_storage], [C], {"transA": False, "transB": True})
    torch.cuda.synchronize()

    # We want C = A * (B_storage)^T, but B_storage == B_logical.t()
    # so (B_storage)^T == B_logical
    ref = (A.float() @ B_logical.float()).contiguous()
    check("Gemm(f16 TC, NT via transB=True storage)", C, ref)


@torch.inference_mode()
def test_gemm_f16_tc_non_multiple_of_16(iters=50):
    """
    WMMA tile 16 기준이지만, 커널이 OOB를 0으로 패딩하도록 구현돼 있으면
    16 배수가 아닌 shape도 동작해야 함.
    (성능은 별개)
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
def test_gemm_f16_tc_cudagraph_capture_safe(iters=10, replays=200):
    """
    CUDA Graph capture/replay safety check.
    """
    M, N, K = 256, 256, 256
    A = torch.randn(M, K, device="cuda", dtype=torch.float16).contiguous()
    B = torch.randn(K, N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()

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


# Example ncu:
# ncu --target-processes all --kernel-name "gemm_f16_tc_wmma_kernel" --launch-count 1 --section SpeedOfLight -o gemm_tc python .\test_gemm_tc.py
#
# Detailed:
# ncu --target-processes all --kernel-name "gemm_f16_tc_wmma_kernel" --launch-count 1 \
#     --section SpeedOfLight --section MemoryWorkloadAnalysis --section LaunchStats --page details \
#     -o gemm_tc_detail python .\test_gemm_tc.py
#
#
# ncu 에서 tensor core 사용 여부 확인
# 
# ncu --target-processes all --kernel-name "gemm_f16_tc_wmma_kernel" --launch-count 1     --section ComputeWorkloadAnalysis --section SpeedOfLight --page details     -o gemm_tc_compute python .\test_gemm_tc.py
