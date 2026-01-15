# examples/python/python_binding_test/v2_ops_unit_smoke.py
from __future__ import annotations

import os
import sys
from pathlib import Path
import random

import torch


# --- auto PYTHONPATH bootstrap ---
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]               # .../ai-compiler-framework (repo root)
EX_PY = ROOT / "examples" / "python" # contains aicf_fw
BUILD_PY = ROOT / "build" / "python" # contains aicf_cuda (built extension)

for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
# --------------------------------

from aicf_fw.backend import set_backend, get_backend
from aicf_fw.backend.aicf_backend import AICFBackend  # 네 프로젝트에 맞춰 경로/이름만 조정


def tf32_off_print():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    print("[torch] allow_tf32.matmul =", torch.backends.cuda.matmul.allow_tf32)
    print("[torch] allow_tf32.cudnn  =", torch.backends.cudnn.allow_tf32)


def mdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float):
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        raise AssertionError(f"[FAIL] {name}: maxdiff={mdiff(got, ref)} atol={atol} rtol={rtol}")


def make_strided_view_2d(x: torch.Tensor) -> torch.Tensor:
    """
    (Stage6 기준) strided 입력도 때려보고 싶을 때 쓰는 helper.
    단, bias_add/relu/copy는 현재 backend 제약상 contig-only로 테스트한다.
    """
    # (B+2, D+2)에서 가운데 view: stride가 (D+2, 1)로 strided
    B, D = x.shape
    big = torch.empty((B + 2, D + 2), device=x.device, dtype=x.dtype)
    big[1:1+B, 1:1+D].copy_(x)
    return big[1:1+B, 1:1+D]


# ----------------------------
# GEMM: 기존처럼 강하게 (views/transposes)
# ----------------------------
def _gemm_ref(A: torch.Tensor, B: torch.Tensor, *, transA: bool, transB: bool) -> torch.Tensor:
    a = A.t() if transA else A
    b = B.t() if transB else B
    return a @ b


def _make_view_2d(x: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "contig":
        return x
    if mode == "t":
        return x.t()
    if mode == "narrow":
        r1 = max(2, x.shape[0] - 1)
        c1 = max(2, x.shape[1] - 1)
        return x[:r1, :c1]
    raise ValueError(mode)


def _alloc_C(M: int, N: int, *, dtype: torch.dtype, device: torch.device, require_contig: bool) -> torch.Tensor:
    if require_contig:
        return torch.empty((M, N), device=device, dtype=dtype)
    big = torch.empty((M + 2, N + 2), device=device, dtype=dtype)
    return big[1:1+M, 1:1+N]


def test_gemm(bk, device):
    print("\n== test_gemm ==")
    torch.manual_seed(0)
    random.seed(0)

    sizes = [(7, 9, 11), (16, 16, 16), (33, 17, 29)]
    trans_flags = [(False, False), (False, True), (True, False), (True, True)]
    views = ["contig", "t", "narrow"]
    dtypes = [torch.float32, torch.float16]

    ok = skip = 0
    for (M, K, N) in sizes:
        for (ta, tb) in trans_flags:
            for dt in dtypes:
                for vA in views:
                    for vB in views:
                        A0 = torch.randn((M, K), device=device, dtype=dt)
                        B0 = torch.randn((K, N), device=device, dtype=dt)
                        A = _make_view_2d(A0, vA)
                        B = _make_view_2d(B0, vB)

                        A_log = A.t() if ta else A
                        B_log = B.t() if tb else B
                        if A_log.shape[1] != B_log.shape[0]:
                            skip += 1
                            continue

                        MM = A_log.shape[0]
                        NN = B_log.shape[1]
                        require_contig_C = (dt == torch.float16)
                        C = _alloc_C(MM, NN, dtype=dt, device=device, require_contig=require_contig_C)

                        ref = _gemm_ref(A, B, transA=ta, transB=tb).to(dt)
                        bk.op_call_out("gemm", [A, B], [C], {"transA": bool(ta), "transB": bool(tb)})

                        if dt == torch.float32:
                            assert_close(f"gemm f32 MKN={M},{K},{N} ta={ta} tb={tb} vA={vA} vB={vB}",
                                         C, ref, atol=1e-4, rtol=1e-4)
                        else:
                            assert_close(f"gemm f16 MKN={M},{K},{N} ta={ta} tb={tb} vA={vA} vB={vB}",
                                         C, ref, atol=3e-2, rtol=3e-2)
                        ok += 1

    print(f"OK gemm (OK={ok} SKIP={skip})")


# ----------------------------
# Stage6-constraint ops: contig-only
# ----------------------------
def test_bias_add_stage6(bk, device):
    """
    Stage6에서는 보통 gemm output(contig)에 bias를 in-place로 더함.
    => contig + in-place 형태만 강하게 검증.
    """
    print("\n== test_bias_add (stage6 contig/in-place only) ==")
    torch.manual_seed(1)

    B, D = 64, 8
    x0 = torch.randn(B, D, device=device, dtype=torch.float32).contiguous()
    b = torch.randn(D, device=device, dtype=torch.float32).contiguous()

    x = x0.clone()
    bk.op_call_out("bias_add", [x, b], [x], {})  # in-place
    ref = x0 + b
    assert_close("bias_add(in-place, contig)", x, ref, atol=1e-6, rtol=1e-6)
    print("OK bias_add (stage6)")


def test_relu_stage6(bk, device):
    """
    Stage6에서 relu 입력은 contig(static)인 경우가 대부분.
    => contig-only로 검증.
    """
    print("\n== test_relu (stage6 contig only) ==")
    torch.manual_seed(2)
    x = torch.randn(64, 8, device=device, dtype=torch.float32).contiguous()
    out = torch.empty_like(x)

    bk.op_call_out("relu", [x], [out], {})
    ref = torch.relu(x)
    assert_close("relu(contig)", out, ref, atol=1e-6, rtol=1e-6)
    print("OK relu (stage6)")


def test_copy_stage6(bk, device):
    """
    Stage6에서 save/copy는 보통 (contig -> contig) 혹은 (contig static reuse) 패턴.
    => contig->contig 만 검증.
    """
    print("\n== test_copy (stage6 contig->contig) ==")
    torch.manual_seed(3)
    x = torch.randn(64, 8, device=device, dtype=torch.float32).contiguous()
    out = torch.empty_like(x)

    bk.op_call_out("copy", [x], [out], {})
    assert_close("copy(contig->contig)", out, x, atol=0.0, rtol=0.0)
    print("OK copy (stage6)")


# ----------------------------
# Already-passed ops (keep)
# ----------------------------
def test_mse_grad(bk, device):
    print("\n== test_mse_grad ==")
    torch.manual_seed(4)
    x = torch.randn(64, 8, device=device, dtype=torch.float32)
    t = torch.randn(64, 8, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    bk.op_call_out("mse_grad", [x, t], [out], {})
    ref = 2.0 * (x - t) / (64 * 8)
    assert_close("mse_grad", out, ref, atol=1e-6, rtol=1e-6)
    print("OK mse_grad")


def test_reduce_sum(bk, device):
    print("\n== test_reduce_sum ==")
    torch.manual_seed(5)
    x = torch.randn(64, 8, device=device, dtype=torch.float32)
    out = torch.empty((8,), device=device, dtype=torch.float32)

    bk.op_call_out("reduce_sum", [x], [out], {"axis": 0, "keepdim": False})
    ref = x.sum(dim=0)
    assert_close("reduce_sum(axis=0)", out, ref, atol=1e-6, rtol=1e-6)
    print("OK reduce_sum")


def test_relu_bwd(bk, device):
    print("\n== test_relu_bwd ==")
    torch.manual_seed(6)
    dy = torch.randn(64, 8, device=device, dtype=torch.float32)
    saved = torch.randn(64, 8, device=device, dtype=torch.float32)
    out = torch.empty_like(dy)

    bk.op_call_out("relu_bwd", [dy, saved], [out], {})
    ref = dy * (saved > 0).to(torch.float32)
    assert_close("relu_bwd", out, ref, atol=1e-6, rtol=1e-6)
    print("OK relu_bwd")


def main():
    tf32_off_print()

    device = torch.device("cuda:0")

    set_backend(AICFBackend())
    bk = get_backend()

    # Strong op-by-op correctness under "Stage6 constraints"
    test_gemm(bk, device)

    # Stage6 실제 패턴 기반: contig-only
    test_bias_add_stage6(bk, device)
    test_relu_stage6(bk, device)
    test_copy_stage6(bk, device)

    # Others (already passed)
    test_mse_grad(bk, device)
    test_reduce_sum(bk, device)
    test_relu_bwd(bk, device)

    print("\nALL OPS PASSED (Stage6 constraints)")


if __name__ == "__main__":
    main()
