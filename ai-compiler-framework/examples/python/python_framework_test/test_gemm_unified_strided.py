# examples/python/python_framework_test/test_gemm_unified_strided.py
from __future__ import annotations

import os
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------
# Make sure we can import aicf_fw when running from repo root / anywhere
# ---------------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]              # .../examples/python
REPO_ROOT = THIS.parents[3]                # .../ai-compiler-framework (adjust if needed)

if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.backend.aicf_backend import AICFBackend  # noqa: E402
from aicf_fw.backend import set_backend  # noqa: E402


def _stats(name: str, y: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> None:
    diff = (y - ref).abs()
    max_abs = float(diff.max().item())
    # rel error: |a-b| / max(|ref|, eps)
    eps = 1e-12
    max_rel = float((diff / torch.clamp(ref.abs(), min=eps)).max().item())
    ok = torch.allclose(y, ref, atol=atol, rtol=rtol)
    print(f"[{name}] allclose={ok} max_abs={max_abs:.6e} max_rel={max_rel:.6e}")
    if not ok:
        raise AssertionError(f"{name} failed: max_abs={max_abs} max_rel={max_rel}")


def aicf_gemm(A: torch.Tensor, B: torch.Tensor, transA: bool, transB: bool) -> torch.Tensor:
    backend = AICFBackend()
    set_backend(backend)
    return backend.op_call("gemm", [A, B], {"transA": transA, "transB": transB})


def make_strided_view_A(M: int, K: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    """
    Create a non-contiguous view with positive strides.
    - We use a bigger contiguous buffer then slice/narrow to get non-contig view.
    """
    base = torch.randn(M, K + 7, device=device, dtype=dtype)  # extra cols
    A = base[:, :K]  # view, still contiguous in many cases; to force non-contig, take transpose view later
    # also make a strided view via transpose (gives stride (1, leading_dim))
    # but we return the base contig; caller will apply .t() etc to create trans views
    return A


def make_strided_view_B(K: int, N: int, device: str, dtype: torch.dtype) -> torch.Tensor:
    base = torch.randn(K + 9, N, device=device, dtype=dtype)  # extra rows
    B = base[:K, :]  # view
    return B


def torch_ref(A: torch.Tensor, B: torch.Tensor, transA: bool, transB: bool) -> torch.Tensor:
    Aop = A.t() if transA else A
    Bop = B.t() if transB else B
    return Aop @ Bop


def run_f32_suite(device: str = "cuda") -> None:
    torch.manual_seed(0)
    M, K, N = 64, 48, 80

    # Build with intentional views.
    A0 = make_strided_view_A(M, K, device, torch.float32)
    B0 = make_strided_view_B(K, N, device, torch.float32)

    # Force some non-contig cases by transposing views without .contiguous()
    A_T_view = A0.t()          # shape (K,M), stride likely (1, ld)
    B_T_view = B0.t()          # shape (N,K)
    print("A_T_view contig?", A_T_view.is_contiguous(), "stride", tuple(A_T_view.stride()))
    print("B_T_view contig?", B_T_view.is_contiguous(), "stride", tuple(B_T_view.stride()))

    cases = [
        ("f32_NN", A0, B0, False, False),
        ("f32_TN", A_T_view, B0, True,  False),
        ("f32_NT", A0, B_T_view, False, True),
        ("f32_TT", A_T_view, B_T_view, True,  True),
    ]

    for name, A, B, tA, tB in cases:
        y = aicf_gemm(A, B, tA, tB)
        ref = torch_ref(A, B, tA, tB)
        # f32 should match exactly for naive implementation (atol/rtol 0 ok in practice)
        _stats(name, y, ref, atol=0.0, rtol=0.0)


def run_f16_suite(device: str = "cuda") -> None:
    torch.manual_seed(1)
    M, K, N = 64, 64, 64  # WMMA likes multiples of 16; but we still allow any via padding in kernel

    A0 = make_strided_view_A(M, K, device, torch.float16)
    B0 = make_strided_view_B(K, N, device, torch.float16)

    # non-contig transpose views
    B_T_view = B0.t()
    A_T_view = A0.t()
    print("B16_T_view contig?", B_T_view.is_contiguous(), "stride", tuple(B_T_view.stride()))

    # Representative cases; add TN/TT if you want
    cases = [
        ("f16_NN_vs_fp32ref", A0, B0, False, False),
        ("f16_NT_vs_fp32ref", A0, B_T_view, False, True),
        # ("f16_TN_vs_fp32ref", A_T_view, B0, True, False),
        # ("f16_TT_vs_fp32ref", A_T_view, B_T_view, True, True),
    ]

    for name, A, B, tA, tB in cases:
        y = aicf_gemm(A, B, tA, tB)
        # reference in fp32
        ref = torch_ref(A.float(), B.float(), tA, tB)
        y32 = y.float()
        # Reasonable tolerance for f16 TC with float acc -> half store
        _stats(name, y32, ref, atol=5e-3, rtol=5e-3)


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.cuda.synchronize()

    run_f32_suite("cuda")
    run_f16_suite("cuda")

    torch.cuda.synchronize()
    print("OK")


if __name__ == "__main__":
    main()

# ncu --target-processes all --launch-count 50 -o tmp_list   python .\test_gemm_unified_strided.py

