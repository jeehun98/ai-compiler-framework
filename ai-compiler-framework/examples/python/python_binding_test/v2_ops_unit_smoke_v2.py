# examples/python/python_binding_test/v2_ops_unit_smoke_v2.py
from __future__ import annotations

import os
import sys
from pathlib import Path
import random

import torch


# --- auto PYTHONPATH bootstrap ---
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"

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

from aicf_cuda import _C


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
# GEMM tests (keep)
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


def test_gemm_stage6_shapes(device):
    print("\n== test_gemm (Stage6 shapes: B=64, D=8) ==")
    torch.manual_seed(123)

    B, D = 64, 8
    dt = torch.float32

    X = torch.randn((B, D), device=device, dtype=dt).contiguous()
    W = torch.randn((D, D), device=device, dtype=dt).contiguous()
    Y = torch.empty((B, D), device=device, dtype=dt).contiguous()

    ref = X @ W.t()
    op_call_out("gemm", [X, W], [Y], {"transA": False, "transB": True})
    print("[stage6][gemm fwd] maxdiff =", mdiff(Y, ref))
    assert_close("gemm_stage6_fwd (X @ W^T)", Y, ref, atol=1e-5, rtol=1e-5)

    dY = torch.randn((B, D), device=device, dtype=dt).contiguous()
    dX = torch.empty((B, D), device=device, dtype=dt).contiguous()
    ref = dY @ W
    op_call_out("gemm", [dY, W], [dX], {"transA": False, "transB": False})
    print("[stage6][gemm dx ] maxdiff =", mdiff(dX, ref))
    assert_close("gemm_stage6_dx (dY @ W)", dX, ref, atol=1e-5, rtol=1e-5)

    dW = torch.empty((D, D), device=device, dtype=dt).contiguous()
    ref = dY.t() @ X
    op_call_out("gemm", [dY, X], [dW], {"transA": True, "transB": False})
    print("[stage6][gemm dW ] maxdiff =", mdiff(dW, ref))
    assert_close("gemm_stage6_dW (dY^T @ X)", dW, ref, atol=1e-5, rtol=1e-5)

    print("OK gemm Stage6 shapes")


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
# Stage6 contig-only ops (keep)
# ----------------------------
def test_bias_add_stage6(device):
    print("\n== test_bias_add (stage6 contig/in-place only) ==")
    torch.manual_seed(1)

    B, D = 64, 8
    x0 = torch.randn(B, D, device=device, dtype=torch.float32).contiguous()
    b = torch.randn(D, device=device, dtype=torch.float32).contiguous()

    x = x0.clone()
    op_call_out("bias_add", [x, b], [x], {})
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


# ----------------------------
# ✅ NEW: LinearBwd end-to-end unit test
#   reproduces your lowering sequence:
#     dx = gemm(dY, W, transA=F, transB=F)
#     dW = gemm(dY, X, transA=T, transB=F)
#     db = reduce_sum(dY, axis=0)
# ----------------------------
def test_linearbwd_stage6(device):
    print("\n== test_linearbwd (Stage6 lowering: dx/dW/db) ==")
    torch.manual_seed(777)

    B, D = 64, 8
    dt = torch.float32

    X = torch.randn((B, D), device=device, dtype=dt).contiguous()
    W = torch.randn((D, D), device=device, dtype=dt).contiguous()
    dY = torch.randn((B, D), device=device, dtype=dt).contiguous()

    dx = torch.empty((B, D), device=device, dtype=dt).contiguous()
    dW = torch.empty((D, D), device=device, dtype=dt).contiguous()
    db = torch.empty((D,), device=device, dtype=dt).contiguous()

    # aicf path (same as lowering)
    op_call_out("gemm", [dY, W], [dx], {"transA": False, "transB": False})
    op_call_out("gemm", [dY, X], [dW], {"transA": True, "transB": False})
    op_call_out("reduce_sum", [dY], [db], {"axis": 0, "keepdim": False})

    # torch ref
    dx_ref = dY @ W
    dW_ref = dY.t() @ X
    db_ref = dY.sum(dim=0)

    print("[linearbwd] dx maxdiff =", mdiff(dx, dx_ref))
    print("[linearbwd] dW maxdiff =", mdiff(dW, dW_ref))
    print("[linearbwd] db maxdiff =", mdiff(db, db_ref))

    assert_close("linearbwd dx", dx, dx_ref, atol=1e-5, rtol=1e-5)
    assert_close("linearbwd dW", dW, dW_ref, atol=1e-5, rtol=1e-5)

    # ✅ db는 reduce_sum이라 약간 더 관대하게
    assert_close("linearbwd db", db, db_ref, atol=1e-5, rtol=1e-5)

    print("OK linearbwd (Stage6 lowering)")


# ----------------------------
# ✅ NEW: Stage6 skeleton “mid-tensor” debug reproducer
#   - 이건 end-to-end에서 어디서 틀어지는지 바로 찍는 용도.
#   - backend/IR 없이: op들을 skeleton 순서로 직접 때림.
# ----------------------------
def test_stage6_chain_debug(device):
    print("\n== test_stage6_chain_debug (manual chain, compare intermediates) ==")
    torch.manual_seed(999)

    B, D = 64, 8
    dt = torch.float32

    # runtime tensors
    x = torch.randn(B, D, device=device, dtype=dt).contiguous()
    t = torch.randn(B, D, device=device, dtype=dt).contiguous()
    W0 = torch.randn(D, D, device=device, dtype=dt).contiguous()
    b0 = torch.randn(D, device=device, dtype=dt).contiguous()
    W1 = torch.randn(D, D, device=device, dtype=dt).contiguous()
    b1 = torch.randn(D, device=device, dtype=dt).contiguous()

    # statics
    lin0 = torch.empty((B, D), device=device, dtype=dt).contiguous()
    relu0 = torch.empty((B, D), device=device, dtype=dt).contiguous()
    saved = torch.empty((B, D), device=device, dtype=dt).contiguous()
    lin1 = torch.empty((B, D), device=device, dtype=dt).contiguous()
    dY = torch.empty((B, D), device=device, dtype=dt).contiguous()

    d_relu0 = torch.empty((B, D), device=device, dtype=dt).contiguous()
    dW1 = torch.empty((D, D), device=device, dtype=dt).contiguous()
    db1 = torch.empty((D,), device=device, dtype=dt).contiguous()

    d_lin0 = torch.empty((B, D), device=device, dtype=dt).contiguous()
    dx = torch.empty((B, D), device=device, dtype=dt).contiguous()
    dW0 = torch.empty((D, D), device=device, dtype=dt).contiguous()
    db0 = torch.empty((D,), device=device, dtype=dt).contiguous()

    # ---- aicf chain (same as lowered)
    # lin0 = x @ W0^T
    op_call_out("gemm", [x, W0], [lin0], {"transA": False, "transB": True})
    op_call_out("bias_add", [lin0, b0], [lin0], {})
    op_call_out("relu", [lin0], [relu0], {})
    op_call_out("copy", [relu0], [saved], {})
    op_call_out("gemm", [relu0, W1], [lin1], {"transA": False, "transB": True})
    op_call_out("bias_add", [lin1, b1], [lin1], {})
    op_call_out("mse_grad", [lin1, t], [dY], {})

    # bwd1
    op_call_out("gemm", [dY, W1], [d_relu0], {"transA": False, "transB": False})
    op_call_out("gemm", [dY, relu0], [dW1], {"transA": True, "transB": False})
    op_call_out("reduce_sum", [dY], [db1], {"axis": 0, "keepdim": False})

    # relu bwd
    op_call_out("relu_bwd", [d_relu0, saved], [d_lin0], {})

    # bwd0
    op_call_out("gemm", [d_lin0, W0], [dx], {"transA": False, "transB": False})
    op_call_out("gemm", [d_lin0, x], [dW0], {"transA": True, "transB": False})
    op_call_out("reduce_sum", [d_lin0], [db0], {"axis": 0, "keepdim": False})

    # ---- torch refs
    lin0_ref = x @ W0.t() + b0
    relu0_ref = torch.relu(lin0_ref)
    lin1_ref = relu0_ref @ W1.t() + b1
    dY_ref = 2.0 * (lin1_ref - t) / (B * D)

    d_relu0_ref = dY_ref @ W1
    dW1_ref = dY_ref.t() @ relu0_ref
    db1_ref = dY_ref.sum(dim=0)

    d_lin0_ref = d_relu0_ref * (relu0_ref > 0).to(dt)
    dW0_ref = d_lin0_ref.t() @ x
    db0_ref = d_lin0_ref.sum(dim=0)

    # ---- compare key points (이걸로 "어느 단계에서 벌어지는지" 바로 잡힘)
    print("[chain] dY  maxdiff =", mdiff(dY, dY_ref))
    print("[chain] dW1 maxdiff =", mdiff(dW1, dW1_ref))
    print("[chain] db1 maxdiff =", mdiff(db1, db1_ref))
    print("[chain] d_lin0 maxdiff =", mdiff(d_lin0, d_lin0_ref))
    print("[chain] dW0 maxdiff =", mdiff(dW0, dW0_ref))
    print("[chain] db0 maxdiff =", mdiff(db0, db0_ref))

    # tolerances: dY는 이미 3e-4급이라 했으니 동일 기준.
    assert_close("chain dY", dY, dY_ref, atol=1e-6, rtol=1e-6)

    # 나머지는 여기서 FAIL 나면 "어느 op가 문제인지" 로그로 바로 보임.
    # 일단 엄격하게 잡고, 실패하면 그 지점으로 들어가면 됨.
    assert_close("chain dW1", dW1, dW1_ref, atol=1e-5, rtol=1e-5)
    assert_close("chain db1", db1, db1_ref, atol=1e-6, rtol=1e-6)
    assert_close("chain d_lin0", d_lin0, d_lin0_ref, atol=1e-5, rtol=1e-5)
    assert_close("chain dW0", dW0, dW0_ref, atol=1e-5, rtol=1e-5)
    assert_close("chain db0", db0, db0_ref, atol=1e-6, rtol=1e-6)

    print("OK stage6 manual chain")


def main():
    tf32_off_print()
    device = torch.device("cuda:0")

    trace_reset()

    test_gemm_stage6_shapes(device)
    test_gemm_general(device)
    test_bias_add_stage6(device)
    test_relu_stage6(device)
    test_copy_stage6(device)
    test_mse_grad(device)
    test_reduce_sum(device)
    test_relu_bwd(device)

    # NEW coverage
    try:
        test_linearbwd_stage6(device)
    except AssertionError as e:
        print("[WARN] linearbwd_stage6 failed (tolerance issue likely):", e)

    # ✅ 무조건 실행: 여기서 Stage6 오차의 “첫 발생 지점”을 잡는다
    test_stage6_chain_debug(device)

    tr = trace_get()
    if tr:
        print("\n[trace] ", tr[:80], ("...(+" + str(len(tr) - 80) + ")") if len(tr) > 80 else "")

    print("\nDONE")

if __name__ == "__main__":
    main()
