from __future__ import annotations

import sys
from pathlib import Path
import struct
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import _C


# -----------------------------------------------------------------------------
# Attr packers
# launcher.cu 기준:
#   schema_id==0: default transA=0, transB=0, relu=1 (Bias+ReLU)
#   schema_id=='GPEL': <iii> (transA, transB, relu)
# NOTE: _C.op_call 의 attr_bytes 를 AttrBlob 로 싸는 쪽이 C++에 있다면,
#       여기 pack bytes 는 GemmEpilogueAttrV0 payload(12B)만 넘기면 됨.
# -----------------------------------------------------------------------------
def pack_gemm_epilogue(trans_a: int = 0, trans_b: int = 0, relu: int = 1) -> bytes:
    return struct.pack("<iii", int(trans_a), int(trans_b), int(relu))


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def meanabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().mean().item())


def assert_close(a: torch.Tensor, b: torch.Tensor, atol: float, rtol: float, tag: str):
    diff = (a - b).abs()
    maxd = float(diff.max().item())
    meand = float(diff.mean().item())
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if not ok:
        raise AssertionError(f"[{tag}] NOT CLOSE: max={maxd:.3e} mean={meand:.3e} atol={atol} rtol={rtol}")
    print(f"[{tag}] ok: max={maxd:.3e} mean={meand:.3e} (atol={atol}, rtol={rtol})")


def sync():
    torch.cuda.synchronize()


def now_ms() -> float:
    return time.perf_counter() * 1e3


def make_noncontig_2d(base: torch.Tensor) -> torch.Tensor:
    # Make a non-contiguous view with positive strides (slice)
    # e.g., take every other column then every other row (still positive stride)
    x = base[:, ::2]
    x = x[::2, :]
    return x


def make_noncontig_1d(base: torch.Tensor) -> torch.Tensor:
    x = base[::2]
    return x


def make_A_B(M: int, K: int, N: int, ta: bool, tb: bool, dtype, device: str = "cuda:0"):
    device = torch.device(device)

    # Create base tensors so that after applying transpose flags,
    # logical A is (M,K) and logical B is (K,N).
    if not ta:
        A = torch.randn(M, K, device=device, dtype=dtype).contiguous()
    else:
        A = torch.randn(K, M, device=device, dtype=dtype).contiguous()

    if not tb:
        B = torch.randn(K, N, device=device, dtype=dtype).contiguous()
    else:
        B = torch.randn(N, K, device=device, dtype=dtype).contiguous()

    return A, B


def torch_gemm_epilogue_ref(
    A: torch.Tensor,
    B: torch.Tensor,
    bias: torch.Tensor,
    ta: bool,
    tb: bool,
    relu: bool,
):
    A2 = A.t() if ta else A
    B2 = B.t() if tb else B
    Y = A2 @ B2
    # bias: (N,)
    Y = Y + bias
    if relu:
        Y = torch.relu(Y)
    return Y


def call_gemm_epilogue(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor, C: torch.Tensor, ta: bool, tb: bool, relu: bool):
    _C.op_call(
        int(_C.OpKind.GemmEpilogue),
        [A, B, bias],
        [C],
        0,
        pack_gemm_epilogue(int(ta), int(tb), int(relu)),
        0,
    )


# -----------------------------------------------------------------------------
# Test cases
# -----------------------------------------------------------------------------
@dataclass
class Case:
    name: str
    M: int
    K: int
    N: int
    ta: bool
    tb: bool
    relu: bool


def run_f32_case(case: Case, atol: float = 1e-4, rtol: float = 1e-4, allow_strided: bool = True) -> Tuple[float, float]:
    A, B = make_A_B(case.M, case.K, case.N, case.ta, case.tb, torch.float32)

    bias = torch.randn(case.N, device="cuda", dtype=torch.float32).contiguous()
    C = torch.empty(case.M, case.N, device="cuda", dtype=torch.float32).contiguous()

    # Optional: make strided inputs/outputs to stress stride path (still positive strides)
    if allow_strided:
        A = make_noncontig_2d(A)  # positive strides, smaller shape now -> adjust logical dims
        # Rebuild B/C/bias to match new logical sizes after slicing
        # We want logical A2@(B2) shape = (M2,N2)
        A2 = A.t() if case.ta else A
        M2, K2 = A2.shape
        # build B to match K2,N2
        N2 = case.N  # keep N fixed for bias tests
        if not case.tb:
            B = torch.randn(K2, N2, device="cuda", dtype=torch.float32).contiguous()
        else:
            B = torch.randn(N2, K2, device="cuda", dtype=torch.float32).contiguous()
        bias = torch.randn(N2, device="cuda", dtype=torch.float32).contiguous()
        C = torch.empty(M2, N2, device="cuda", dtype=torch.float32).contiguous()

    C_ref = torch_gemm_epilogue_ref(A, B, bias, case.ta, case.tb, case.relu)

    call_gemm_epilogue(A, B, bias, C, case.ta, case.tb, case.relu)
    sync()

    maxd = maxabs_delta(C, C_ref)
    meand = meanabs_delta(C, C_ref)
    assert_close(C, C_ref, atol=atol, rtol=rtol, tag=f"F32/{case.name}")
    return maxd, meand


def run_f16_tc_case(case: Case, atol: float = 5e-2, rtol: float = 5e-2) -> Tuple[float, float]:
    # TC path: launcher가 C contiguous row-major 요구
    A, B = make_A_B(case.M, case.K, case.N, case.ta, case.tb, torch.float16)

    # NOTE(v0): f16 path bias dtype == f16
    bias = torch.randn(case.N, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(case.M, case.N, device="cuda", dtype=torch.float16).contiguous()

    # ref: float compute -> half
    C_ref = torch_gemm_epilogue_ref(A.float(), B.float(), bias.float(), case.ta, case.tb, case.relu).half()

    call_gemm_epilogue(A, B, bias, C, case.ta, case.tb, case.relu)
    sync()

    maxd = maxabs_delta(C.float(), C_ref.float())
    meand = meanabs_delta(C.float(), C_ref.float())
    assert_close(C.float(), C_ref.float(), atol=atol, rtol=rtol, tag=f"F16-TC/{case.name}")
    return maxd, meand


def bench(case: Case, dtype: torch.dtype, iters: int = 200, warmup: int = 20) -> float:
    ta, tb, relu = case.ta, case.tb, case.relu
    if dtype == torch.float32:
        A, B = make_A_B(case.M, case.K, case.N, ta, tb, torch.float32)
        bias = torch.randn(case.N, device="cuda", dtype=torch.float32).contiguous()
        C = torch.empty(case.M, case.N, device="cuda", dtype=torch.float32).contiguous()
        ref = None
    elif dtype == torch.float16:
        A, B = make_A_B(case.M, case.K, case.N, ta, tb, torch.float16)
        bias = torch.randn(case.N, device="cuda", dtype=torch.float16).contiguous()
        C = torch.empty(case.M, case.N, device="cuda", dtype=torch.float16).contiguous()
        ref = None
    else:
        raise ValueError(dtype)

    # warmup
    for _ in range(warmup):
        call_gemm_epilogue(A, B, bias, C, ta, tb, relu)
    sync()

    t0 = now_ms()
    for _ in range(iters):
        call_gemm_epilogue(A, B, bias, C, ta, tb, relu)
    sync()
    t1 = now_ms()

    return (t1 - t0) / iters


# -----------------------------------------------------------------------------
# NEG tests
# -----------------------------------------------------------------------------
def expect_fail(fn, tag: str):
    try:
        fn()
        raise AssertionError(f"[{tag}] unexpected OK")
    except RuntimeError as e:
        print(f"[{tag}] ok:", str(e).splitlines()[0])
    except Exception as e:
        # still a fail, but different exception type
        print(f"[{tag}] raised non-RuntimeError:", type(e).__name__, str(e).splitlines()[0])


def neg_wrong_c_shape():
    A, B = make_A_B(8, 4, 7, False, False, torch.float32)
    bias = torch.randn(7, device="cuda", dtype=torch.float32).contiguous()
    C_bad = torch.empty(8, 8, device="cuda", dtype=torch.float32).contiguous()
    call_gemm_epilogue(A, B, bias, C_bad, False, False, True)


def neg_wrong_bias_len():
    A, B = make_A_B(8, 4, 7, False, False, torch.float32)
    C = torch.empty(8, 7, device="cuda", dtype=torch.float32).contiguous()
    bias_bad = torch.randn(8, device="cuda", dtype=torch.float32).contiguous()
    call_gemm_epilogue(A, B, bias_bad, C, False, False, True)


def neg_wrong_dtype_mix():
    A16, B16 = make_A_B(16, 16, 16, False, False, torch.float16)
    bias16 = torch.randn(16, device="cuda", dtype=torch.float16).contiguous()
    C32 = torch.empty(16, 16, device="cuda", dtype=torch.float32).contiguous()
    call_gemm_epilogue(A16, B16, bias16, C32, False, False, True)


def neg_noncontig_c_for_tc():
    # should select f32 kernel if dtype mismatch; here force half path and violate contig C
    A, B = make_A_B(64, 64, 64, False, False, torch.float16)
    bias = torch.randn(64, device="cuda", dtype=torch.float16).contiguous()
    C = torch.empty(64, 64, device="cuda", dtype=torch.float16).t()  # non-contig
    call_gemm_epilogue(A, B, bias, C, False, False, True)


def neg_bias_stride_zero_like():
    # not easy to create stride 0 in torch 1D unless expand
    A, B = make_A_B(16, 16, 16, False, False, torch.float32)
    bias0 = torch.randn(1, device="cuda", dtype=torch.float32).expand(16)  # stride 0
    C = torch.empty(16, 16, device="cuda", dtype=torch.float32).contiguous()
    call_gemm_epilogue(A, B, bias0, C, False, False, True)


def run_bwd_f32(M=64, N=80, relu=True):
    device = torch.device("cuda")

    # Forward 결과 Y 가 있다고 가정
    Y = torch.randn(M, N, device=device, dtype=torch.float32).contiguous()
    dY = torch.randn(M, N, device=device, dtype=torch.float32).contiguous()

    dBias = torch.empty(N, device=device, dtype=torch.float32).contiguous()

    # Reference
    if relu:
        mask = (Y > 0).float()
        dBias_ref = (dY * mask).sum(dim=0)
    else:
        dBias_ref = dY.sum(dim=0)

    _C.op_call(
        int(_C.OpKind.GemmEpilogueBwd),
        [dY, Y],
        [dBias],
        0,
        pack_gemm_epilogue(0, 0, int(relu)),
        0,
    )

    torch.cuda.synchronize()

    maxd = maxabs_delta(dBias, dBias_ref)
    print(f"[BWD F32] relu={int(relu)} max|delta|={maxd:.3e}")
    return maxd

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    print("OpKind.GemmEpilogue =", int(_C.OpKind.GemmEpilogue))
    if hasattr(_C.OpKind, "GemmEpilogueBwd"):
        print("OpKind.GemmEpilogueBwd =", int(_C.OpKind.GemmEpilogueBwd))

    # -------------------------
    # Correctness: f32 (small + edge)
    # -------------------------
    cases_f32: List[Case] = []
    for relu in (True, False):
        for ta in (False, True):
            for tb in (False, True):
                cases_f32.append(Case(name=f"M64K48N80_ta{int(ta)}_tb{int(tb)}_r{int(relu)}",
                                      M=64, K=48, N=80, ta=ta, tb=tb, relu=relu))

    # edge sizes (not multiples of 16)
    cases_f32 += [
        Case(name="edge_M1K3N5", M=1,  K=3,  N=5,  ta=False, tb=False, relu=True),
        Case(name="edge_M17K19N23", M=17, K=19, N=23, ta=False, tb=False, relu=False),
        Case(name="edge_M33K7N9_ta1", M=33, K=7, N=9, ta=True, tb=False, relu=True),
        Case(name="edge_M9K33N5_tb1", M=9, K=33, N=5, ta=False, tb=True, relu=False),
    ]

    worst_max = 0.0
    for c in cases_f32:
        maxd, _ = run_f32_case(c, atol=1e-4, rtol=1e-4, allow_strided=False)
        worst_max = max(worst_max, maxd)
    print("[F32] worst max|delta| =", worst_max)

    # f32 strided stress (positive stride only)
    worst_max_strided = 0.0
    for relu in (True, False):
        for ta in (False, True):
            for tb in (False, True):
                c = Case(name=f"strided_ta{int(ta)}_tb{int(tb)}_r{int(relu)}",
                         M=64, K=48, N=80, ta=ta, tb=tb, relu=relu)
                maxd, _ = run_f32_case(c, atol=1e-4, rtol=1e-4, allow_strided=True)
                worst_max_strided = max(worst_max_strided, maxd)
    print("[F32 strided] worst max|delta| =", worst_max_strided)

    # -------------------------
    # Correctness: f16 TC (wmma)
    # -------------------------
    cases_tc: List[Case] = []
    for relu in (True, False):
        for ta in (False, True):
            for tb in (False, True):
                cases_tc.append(Case(name=f"M64K64N64_ta{int(ta)}_tb{int(tb)}_r{int(relu)}",
                                     M=64, K=64, N=64, ta=ta, tb=tb, relu=relu))

    worst_tc = 0.0
    for c in cases_tc:
        maxd, _ = run_f16_tc_case(c, atol=5e-2, rtol=5e-2)
        worst_tc = max(worst_tc, maxd)
    print("[F16-TC] worst max|delta| =", worst_tc)

    # -------------------------
    # NEG tests
    # -------------------------
    expect_fail(neg_wrong_c_shape, "NEG wrong C shape")
    expect_fail(neg_wrong_bias_len, "NEG wrong bias len")
    expect_fail(neg_wrong_dtype_mix, "NEG wrong dtype mix")
    expect_fail(neg_noncontig_c_for_tc, "NEG noncontig C for TC path")
    expect_fail(neg_bias_stride_zero_like, "NEG bias stride 0 (expand)")

    # -------------------------
    # Simple perf smoke (not a benchmark suite)
    # -------------------------
    perf_case_f32 = Case(name="perf_f32", M=256, K=256, N=256, ta=False, tb=False, relu=True)
    perf_case_f16 = Case(name="perf_f16", M=256, K=256, N=256, ta=False, tb=False, relu=True)

    t_f32 = bench(perf_case_f32, torch.float32, iters=200, warmup=50)
    t_f16 = bench(perf_case_f16, torch.float16, iters=200, warmup=50)

    print(f"[PERF] f32 avg ms = {t_f32:.4f}")
    print(f"[PERF] f16 avg ms = {t_f16:.4f}")

    print("\n--- BWD TEST ---")
    run_bwd_f32(64, 80, relu=True)
    run_bwd_f32(64, 80, relu=False)

    print("DONE")


if __name__ == "__main__":
    main()