# examples/python/python_binding_test/v2_ops_unit_smoke.py
from __future__ import annotations

import os
import sys
from pathlib import Path
import random

import torch


# --- auto PYTHONPATH bootstrap ---
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]                # .../ai-compiler-framework (repo root)
EX_PY = ROOT / "examples" / "python"  # contains aicf_fw (optional, not used here)
BUILD_PY = ROOT / "build" / "python"  # contains aicf_cuda (built extension)

for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

if os.name == "nt":
    try:
        os.add_dll_directory(str(BUILD_PY))
        os.add_dll_directory(str(BUILD_PY / "aicf_cuda"))
    except Exception:
        pass
# --------------------------------

from aicf_cuda import _C  # ✅ direct binding


# ----------------------------
# global knobs
# ----------------------------
def tf32_off_print():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass
    print("[torch] allow_tf32.matmul =", torch.backends.cuda.matmul.allow_tf32)
    print("[torch] allow_tf32.cudnn  =", torch.backends.cuda.cudnn.allow_tf32 if hasattr(torch.backends.cuda, "cudnn") else torch.backends.cudnn.allow_tf32)


# ----------------------------
# helpers
# ----------------------------
def mdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def assert_close(name: str, got: torch.Tensor, ref: torch.Tensor, *, atol: float, rtol: float):
    if not torch.allclose(got, ref, atol=atol, rtol=rtol):
        raise AssertionError(f"[FAIL] {name}: maxdiff={mdiff(got, ref)} atol={atol} rtol={rtol}")


# op-name -> OpKind
_OPNAME_TO_KIND = {
    "add": int(_C.OpKind.EltwiseAdd),
    "relu": int(_C.OpKind.EltwiseRelu),
    "gemm": int(_C.OpKind.Gemm),
    "bias_add": int(_C.OpKind.BiasAdd),
    "reduce_sum": int(_C.OpKind.ReduceSum),
    "mse_grad": int(_C.OpKind.MseGrad),
    "relu_bwd": int(_C.OpKind.ReluBwd),
    "copy": int(_C.OpKind.Copy),
    "adam_step": int(_C.OpKind.AdamStep),
    "step_inc": int(_C.OpKind.StepInc),
    "bias_corr": int(_C.OpKind.BiasCorr),
}


def op_call_out(op: str, inputs: list[torch.Tensor], outputs: list[torch.Tensor], attrs: dict):
    k = op.strip().lower()
    if k not in _OPNAME_TO_KIND:
        raise RuntimeError(f"unknown op '{op}' (no OpKind mapping)")
    _C.op_call(int(_OPNAME_TO_KIND[k]), list(inputs), list(outputs), dict(attrs))


def trace_reset():
    fn = getattr(_C, "trace_reset", None)
    if fn is not None:
        fn()


def trace_get() -> list[str]:
    fn = getattr(_C, "trace_get", None)
    if fn is None:
        return []
    return list(fn())


# ----------------------------
# GEMM: helpers
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
    return big[1:1 + M, 1:1 + N]


# ----------------------------
# GEMM: 기존 강한 테스트 (views/transposes)
# ----------------------------
def test_gemm_general(device):
    print("\n== test_gemm (general views/transposes) ==")
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
                        op_call_out("gemm", [A, B], [C], {"transA": bool(ta), "transB": bool(tb)})

                        if dt == torch.float32:
                            assert_close(
                                f"gemm f32 MKN={M},{K},{N} ta={ta} tb={tb} vA={vA} vB={vB}",
                                C, ref, atol=1e-4, rtol=1e-4
                            )
                        else:
                            assert_close(
                                f"gemm f16 MKN={M},{K},{N} ta={ta} tb={tb} vA={vA} vB={vB}",
                                C, ref, atol=3e-2, rtol=3e-2
                            )
                        ok += 1

    print(f"OK gemm general (OK={ok} SKIP={skip})")


# ----------------------------
# ✅ 추가: Stage6 동일 shape/조합 GEMM 재현 테스트
#   - forward:  y = X @ W^T        => gemm(X, W, transB=True)
#   - bwd dX:   dX = dY @ W        => gemm(dY, W, transA=False, transB=False)
#   - bwd dW:   dW = dY^T @ X      => gemm(dY, X, transA=True,  transB=False)
# ----------------------------
def test_gemm_stage6_shapes(device):
    print("\n== test_gemm (Stage6 shapes: B=64, D=8) ==")
    torch.manual_seed(123)

    B, D = 64, 8
    dt = torch.float32

    # 1) forward linear: (B,D) @ (D,D) -> (B,D) with transB=True where W is (D,D) stored as (OUT,IN)
    X = torch.randn((B, D), device=device, dtype=dt).contiguous()
    W = torch.randn((D, D), device=device, dtype=dt).contiguous()
    Y = torch.empty((B, D), device=device, dtype=dt).contiguous()

    ref = X @ W.t()
    op_call_out("gemm", [X, W], [Y], {"transA": False, "transB": True})
    print("[stage6][gemm fwd] maxdiff =", mdiff(Y, ref))
    assert_close("gemm_stage6_fwd (X @ W^T)", Y, ref, atol=1e-5, rtol=1e-5)

    # 2) linear bwd dX: dY @ W -> (B,D)
    dY = torch.randn((B, D), device=device, dtype=dt).contiguous()
    dX = torch.empty((B, D), device=device, dtype=dt).contiguous()
    ref = dY @ W
    op_call_out("gemm", [dY, W], [dX], {"transA": False, "transB": False})
    print("[stage6][gemm dx ] maxdiff =", mdiff(dX, ref))
    assert_close("gemm_stage6_dx (dY @ W)", dX, ref, atol=1e-5, rtol=1e-5)

    # 3) linear bwd dW: dY^T @ X -> (D,D)  (NOTE: your lowering uses gemm(dY, X, transA=True))
    dW = torch.empty((D, D), device=device, dtype=dt).contiguous()
    ref = dY.t() @ X
    op_call_out("gemm", [dY, X], [dW], {"transA": True, "transB": False})
    print("[stage6][gemm dW ] maxdiff =", mdiff(dW, ref))
    assert_close("gemm_stage6_dW (dY^T @ X)", dW, ref, atol=1e-5, rtol=1e-5)

    print("OK gemm Stage6 shapes")


# ----------------------------
# Stage6-constraint ops: contig-only
# ----------------------------
def test_bias_add_stage6(device):
    print("\n== test_bias_add (stage6 contig/in-place only) ==")
    torch.manual_seed(1)

    B, D = 64, 8
    x0 = torch.randn(B, D, device=device, dtype=torch.float32).contiguous()
    b = torch.randn(D, device=device, dtype=torch.float32).contiguous()

    x = x0.clone()
    op_call_out("bias_add", [x, b], [x], {})  # in-place
    ref = x0 + b
    assert_close("bias_add(in-place, contig)", x, ref, atol=1e-6, rtol=1e-6)
    print("OK bias_add (stage6)")


def test_relu_stage6(device):
    print("\n== test_relu (stage6 contig only) ==")
    torch.manual_seed(2)
    x = torch.randn(64, 8, device=device, dtype=torch.float32).contiguous()
    out = torch.empty_like(x)

    op_call_out("relu", [x], [out], {})
    ref = torch.relu(x)
    assert_close("relu(contig)", out, ref, atol=1e-6, rtol=1e-6)
    print("OK relu (stage6)")


def test_copy_stage6(device):
    print("\n== test_copy (stage6 contig->contig) ==")
    torch.manual_seed(3)
    x = torch.randn(64, 8, device=device, dtype=torch.float32).contiguous()
    out = torch.empty_like(x)

    op_call_out("copy", [x], [out], {})
    assert_close("copy(contig->contig)", out, x, atol=0.0, rtol=0.0)
    print("OK copy (stage6)")


# ----------------------------
# Others
# ----------------------------
def test_mse_grad(device):
    print("\n== test_mse_grad ==")
    torch.manual_seed(4)
    x = torch.randn(64, 8, device=device, dtype=torch.float32)
    t = torch.randn(64, 8, device=device, dtype=torch.float32)
    out = torch.empty_like(x)

    op_call_out("mse_grad", [x, t], [out], {})
    ref = 2.0 * (x - t) / (64 * 8)
    assert_close("mse_grad", out, ref, atol=1e-6, rtol=1e-6)
    print("OK mse_grad")


def test_reduce_sum(device):
    print("\n== test_reduce_sum ==")
    torch.manual_seed(5)
    x = torch.randn(64, 8, device=device, dtype=torch.float32)
    out = torch.empty((8,), device=device, dtype=torch.float32)

    op_call_out("reduce_sum", [x], [out], {"axis": 0, "keepdim": False})
    ref = x.sum(dim=0)
    assert_close("reduce_sum(axis=0)", out, ref, atol=1e-6, rtol=1e-6)
    print("OK reduce_sum")


def test_relu_bwd(device):
    print("\n== test_relu_bwd ==")
    torch.manual_seed(6)
    dy = torch.randn(64, 8, device=device, dtype=torch.float32)
    saved = torch.randn(64, 8, device=device, dtype=torch.float32)
    out = torch.empty_like(dy)

    op_call_out("relu_bwd", [dy, saved], [out], {})
    ref = dy * (saved > 0).to(torch.float32)
    assert_close("relu_bwd", out, ref, atol=1e-6, rtol=1e-6)
    print("OK relu_bwd")


def main():
    tf32_off_print()
    device = torch.device("cuda:0")

    trace_reset()

    # ✅ 먼저 Stage6 shape 재현부터 (지금 dW 오차 원인 분리용)
    test_gemm_stage6_shapes(device)

    # 기존 강한 테스트
    test_gemm_general(device)

    # Stage6 실제 패턴 기반: contig-only
    test_bias_add_stage6(device)
    test_relu_stage6(device)
    test_copy_stage6(device)

    # Others
    test_mse_grad(device)
    test_reduce_sum(device)
    test_relu_bwd(device)

    tr = trace_get()
    if tr:
        print("\n[trace] ", tr)

    print("\nALL OPS PASSED (direct aicf_cuda._C op_call)")


if __name__ == "__main__":
    main()
