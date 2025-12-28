# examples/python/python_framework_test/test_gemm_unified_strided_cli.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

# ---------------------------------------------------------------------
# Import path fix: allow running from anywhere
# ---------------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.backend.aicf_backend import AICFBackend  # noqa: E402
from aicf_fw.backend import set_backend  # noqa: E402


def aicf_gemm(backend: AICFBackend, A: torch.Tensor, B: torch.Tensor, transA: bool, transB: bool) -> torch.Tensor:
    return backend.op_call("gemm", [A, B], {"transA": transA, "transB": transB})


def torch_ref(A: torch.Tensor, B: torch.Tensor, transA: bool, transB: bool) -> torch.Tensor:
    Aop = A.t() if transA else A
    Bop = B.t() if transB else B
    return Aop @ Bop


def stats(name: str, y: torch.Tensor, ref: torch.Tensor, atol: float, rtol: float) -> None:
    diff = (y - ref).abs()
    max_abs = float(diff.max().item())
    eps = 1e-12
    max_rel = float((diff / torch.clamp(ref.abs(), min=eps)).max().item())
    ok = torch.allclose(y, ref, atol=atol, rtol=rtol)
    print(f"[{name}] allclose={ok} max_abs={max_abs:.6e} max_rel={max_rel:.6e}")
    if not ok:
        raise AssertionError(f"{name} failed: max_abs={max_abs} max_rel={max_rel}")


def make_inputs(M: int, K: int, N: int, dtype: torch.dtype, device: str, seed: int):
    torch.manual_seed(seed)

    # Make strided-ish A/B via slicing from a larger base, then use transpose views for trans cases
    A_base = torch.randn(M, K + 7, device=device, dtype=dtype)
    B_base = torch.randn(K + 9, N, device=device, dtype=dtype)

    A = A_base[:, :K]       # view
    B = B_base[:K, :]       # view

    A_T = A.t()             # non-contig view (K,M)
    B_T = B.t()             # non-contig view (N,K)

    return A, B, A_T, B_T


def parse_case(case: str):
    c = case.upper()
    if c not in ("NN", "TN", "NT", "TT"):
        raise ValueError("--case must be one of NN/TN/NT/TT")
    transA = c[0] == "T"
    transB = c[1] == "T"
    return transA, transB


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dtype", choices=["f32", "f16"], default="f16")
    ap.add_argument("--case", choices=["NN", "TN", "NT", "TT"], default="NT")
    ap.add_argument("--M", type=int, default=64)
    ap.add_argument("--K", type=int, default=64)
    ap.add_argument("--N", type=int, default=64)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--device", default="cuda")

    # 핵심: NCU에서 torch 커널(ampere_sgemm_*) 안 보이게 제어
    ap.add_argument("--aicf-only", action="store_true",
                    help="If set, do not run torch GEMM ref inside profiled region. (Ref computed after sync.)")
    ap.add_argument("--no-ref", action="store_true",
                    help="If set, skip reference check entirely (kernel launch only).")

    args = ap.parse_args()

    assert torch.cuda.is_available()
    device = args.device

    dtype = torch.float32 if args.dtype == "f32" else torch.float16
    M, K, N = args.M, args.K, args.N

    # shapes for transpose cases:
    # A physical is (M,K), A_T is (K,M). When transA=True, pass A_T as input A with attr transA=True
    # B physical is (K,N), B_T is (N,K). When transB=True, pass B_T as input B with attr transB=True
    A, B, A_T, B_T = make_inputs(M, K, N, dtype, device, seed=args.seed)

    transA, transB = parse_case(args.case)

    A_in = A_T if transA else A
    B_in = B_T if transB else B

    print("A_in contig?", A_in.is_contiguous(), "stride", tuple(A_in.stride()), "shape", tuple(A_in.shape))
    print("B_in contig?", B_in.is_contiguous(), "stride", tuple(B_in.stride()), "shape", tuple(B_in.shape))
    torch.cuda.synchronize()

    backend = AICFBackend()
    set_backend(backend)

    # Optional warmup (kept minimal; comment out if you want literally one launch only)
    # _ = aicf_gemm(backend, A_in, B_in, transA, transB)
    # torch.cuda.synchronize()

    # -------------------------
    # Profile target: AICF GEMM exactly once
    # -------------------------
    y = aicf_gemm(backend, A_in, B_in, transA, transB)
    torch.cuda.synchronize()

    if args.no_ref:
        print("NO_REF (kernel launch only) OK")
        return

    # -------------------------
    # Reference (optionally pushed outside "interesting" region)
    # -------------------------
    if args.aicf_only:
        # compute ref AFTER sync; still in same process, but you will kernel-filter in ncu anyway
        # This just prevents ref from interleaving with the AICF launch timing-wise.
        pass

    if dtype == torch.float32:
        ref = torch_ref(A_in, B_in, transA, transB)
        stats(f"f32_{args.case}", y, ref, atol=0.0, rtol=0.0)
    else:
        # fp32 ref for f16
        ref = torch_ref(A_in.float(), B_in.float(), transA, transB)
        stats(f"f16_{args.case}_vs_fp32ref", y.float(), ref, atol=5e-3, rtol=5e-3)

    print("OK")


if __name__ == "__main__":
    main()

# ncu -f -o gemm_tc_nt   --target-processes all   --launch-count 1   -k "regex:.*gemm_f16_tc_wmma_out_f16_strided_kernel.*"   --section SpeedOfLight --section WarpStateStats --section SchedulerStats   python .\test_gemm_unified_strided_cli.py --dtype f16 --case NT --M 64 --K 64 --N 64 --aicf-only
