# examples/python/test_gemm_trans_strided.py
from __future__ import annotations

import sys
from pathlib import Path

# Make "examples/python" importable (so `import aicf_fw` works)
THIS = Path(__file__).resolve()
PY_ROOT = THIS.parents[1]  # .../examples/python
if str(PY_ROOT) not in sys.path:
    sys.path.insert(0, str(PY_ROOT))

import torch

from aicf_fw.backend.aicf_backend import AICFBackend
from aicf_fw.backend import set_backend


def _aicf_gemm(A: torch.Tensor, B: torch.Tensor, transA: bool, transB: bool) -> torch.Tensor:
    backend = AICFBackend()
    set_backend(backend)
    return backend.op_call("gemm", [A, B], {"transA": transA, "transB": transB})


def _ref_gemm(A: torch.Tensor, B: torch.Tensor, transA: bool, transB: bool) -> torch.Tensor:
    Aop = A.t() if transA else A
    Bop = B.t() if transB else B
    return Aop @ Bop


def _assert_close(name: str, x: torch.Tensor, y: torch.Tensor, atol: float, rtol: float) -> None:
    # tolerate fp16 a bit more
    max_abs = float((x - y).abs().max().item())
    max_rel = float(((x - y).abs() / (y.abs() + 1e-8)).max().item())
    ok = torch.allclose(x, y, atol=atol, rtol=rtol)
    print(f"[{name}] allclose={ok} max_abs={max_abs:.6e} max_rel={max_rel:.6e}")
    if not ok:
        raise AssertionError(f"{name}: mismatch (see max_abs/max_rel above)")


def main():
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    device = "cuda"

    # ----------------------------
    # Case 1: f32 correctness (NN / TN / NT / TT)
    # ----------------------------
    M, K, N = 64, 48, 80
    A = torch.randn(M, K, device=device, dtype=torch.float32)
    B = torch.randn(K, N, device=device, dtype=torch.float32)

    # NN
    C = _aicf_gemm(A, B, False, False)
    Cref = _ref_gemm(A, B, False, False)
    _assert_close("f32_NN", C, Cref, atol=1e-4, rtol=1e-4)

    # TN: A^T @ B  => A is (M,K) but transA=True makes it (K,M); so we need A shaped (K,M)
    # Provide A_T_view as a view (no contiguous) to test stride propagation.
    A_T_view = A.t()  # (K,M), non-contig view
    C = _aicf_gemm(A_T_view, B, True, False)
    Cref = _ref_gemm(A_T_view, B, True, False)
    _assert_close("f32_TN", C, Cref, atol=1e-4, rtol=1e-4)

    # NT: A @ B^T  => provide B_T_view (N,K), transB=True makes it (K,N) logically
    B_T_view = B.t()  # (N,K), non-contig view
    C = _aicf_gemm(A, B_T_view, False, True)
    Cref = _ref_gemm(A, B_T_view, False, True)
    _assert_close("f32_NT", C, Cref, atol=1e-4, rtol=1e-4)

    # TT: A^T @ B^T  => provide both views
    C = _aicf_gemm(A_T_view, B_T_view, True, True)
    Cref = _ref_gemm(A_T_view, B_T_view, True, True)
    _assert_close("f32_TT", C, Cref, atol=1e-4, rtol=1e-4)

    # ----------------------------
    # Case 2: f16 TC path sanity (if your TC variant supports it)
    # ----------------------------
    # Use multiples of 16 to avoid edge complications.
    M, K, N = 64, 64, 64
    A16 = torch.randn(M, K, device=device, dtype=torch.float16)
    B16 = torch.randn(K, N, device=device, dtype=torch.float16)

    # NN
    C16 = _aicf_gemm(A16, B16, False, False)
    C16_ref = _ref_gemm(A16, B16, False, False)

    # fp16 accumulation differences exist; compare to fp32-ref for stricter numeric stability
    C16_ref_f32 = (A16.float() @ B16.float()).half()
    _assert_close("f16_NN_vs_fp32ref", C16, C16_ref_f32, atol=5e-2, rtol=5e-2)

    # NT with view (B^T view)
    B16_T_view = B16.t()  # (N,K), non-contig
    C16 = _aicf_gemm(A16, B16_T_view, False, True)
    C16_ref_f32 = (A16.float() @ B16_T_view.t().float()).half()
    _assert_close("f16_NT_vs_fp32ref", C16, C16_ref_f32, atol=5e-2, rtol=5e-2)

    # ----------------------------
    # Case 3: show that views are truly non-contiguous (debug)
    # ----------------------------
    print("A_T_view contig?", A_T_view.is_contiguous(), "stride", A_T_view.stride())
    print("B_T_view contig?", B_T_view.is_contiguous(), "stride", B_T_view.stride())
    print("B16_T_view contig?", B16_T_view.is_contiguous(), "stride", B16_T_view.stride())

    torch.cuda.synchronize()
    print("OK")


if __name__ == "__main__":
    main()
