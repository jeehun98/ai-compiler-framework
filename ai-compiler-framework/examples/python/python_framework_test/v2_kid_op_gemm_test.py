from __future__ import annotations

import torch

from _test_path_bootstrap import ensure_test_paths, import_cuda_ext
from _kid_op_utils import pack_gemm_attr, launch_by_id, assert_allclose

ensure_test_paths()
_C = import_cuda_ext()


def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def main():
    tf32_off()
    torch.manual_seed(0)
    device = torch.device("cuda:0")
    dtype = torch.float16

    # 작은 크기 + TC 타일 맞춰지는 케이스
    M, K, N = 64, 8, 8

    A = torch.randn(M, K, device=device, dtype=dtype)
    B = torch.randn(N, K, device=device, dtype=dtype)  # W: (N,K) for transB=True 케이스 테스트
    C = torch.empty(M, N, device=device, dtype=dtype)

    # y = A @ B^T   (B is (N,K))
    transA = False
    transB = True

    # reference
    ref = A @ B.t()

    # launch
    kernel_id = "gemm_f16_tc_wmma_out_f16_v0"
    schema_id, attrs = pack_gemm_attr(transA, transB)

    # inputs: (A, B), outputs: (C)
    launch_by_id(kernel_id, _C.OpKind.Gemm, [A, B], [C], schema_id, attrs, stream=0)

    # check
    assert_allclose(C, ref, atol=5e-2, rtol=5e-2, msg="GEMM f16 TC mismatch")
    print("[OK] gemm_f16_tc_wmma_out_f16_v0 transB=True works.")

    # 추가: transA=True 케이스도 한 번 더
    # A2: (K,M) then transA=True => effective (M,K)
    A2 = torch.randn(K, M, device=device, dtype=dtype).contiguous()
    B2 = torch.randn(K, N, device=device, dtype=dtype).contiguous()  # transB=False => (K,N)
    C2 = torch.empty(M, N, device=device, dtype=dtype)

    transA = True
    transB = False
    ref2 = A2.t() @ B2  # (M,K)@(K,N)

    schema_id2, attrs2 = pack_gemm_attr(transA, transB)
    launch_by_id(kernel_id, _C.OpKind.Gemm, [A2, B2], [C2], schema_id2, attrs2, stream=0)

    assert_allclose(C2, ref2, atol=5e-2, rtol=5e-2, msg="GEMM f16 TC mismatch (transA=True)")
    print("[OK] gemm_f16_tc_wmma_out_f16_v0 transA=True works.")

    print("OK")


if __name__ == "__main__":
    main()
