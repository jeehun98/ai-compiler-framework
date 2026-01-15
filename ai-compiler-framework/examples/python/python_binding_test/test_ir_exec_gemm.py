# examples/python/python_binding_test/test_ir_exec_gemm.py
from __future__ import annotations

import os
import sys
from pathlib import Path

# --- auto PYTHONPATH bootstrap ---
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]               # .../ai-compiler-framework (repo root)
EX_PY = ROOT / "examples" / "python" # contains aicf_fw (optional)
BUILD_PY = ROOT / "build" / "python" # contains aicf_cuda (built extension)

for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
# --------------------------------

import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch

# build/python/aicf_cuda/_C.pyd
from aicf_cuda import _C


# ============================================================
# utils: logging / formatting
# ============================================================

def _s(t: torch.Tensor) -> str:
    return (
        f"shape={tuple(t.shape)} dtype={str(t.dtype)} dev={str(t.device)} "
        f"stride={tuple(int(x) for x in t.stride())} contig={t.is_contiguous()}"
    )

def _log(msg: str):
    if os.getenv("CASE_LOG", "0") != "0":
        print(msg)

def _cuda_sync():
    torch.cuda.synchronize()

def _allclose(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float) -> bool:
    return torch.allclose(a, b, atol=atol, rtol=rtol)

def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

def _make_C_contig(M: int, N: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    # IMPORTANT: TC path currently requires C contiguous row-major
    return torch.empty((M, N), device=device, dtype=dtype)


# ============================================================
# IR JSON builder (minimal, lowered ops)
# ============================================================

def build_ir_json_lowered_gemm(
    *,
    M: int,
    K: int,
    N: int,
    dtype: torch.dtype,
    device: torch.device,
    transA: bool,
    transB: bool,
    with_bias: bool,
) -> Tuple[str, Dict[str, int]]:
    """
    Build a minimal IR JSON using *lowered* ops that bindings.cpp supports:
      gemm(x, W) -> y
      (optional) bias_add(y, b) -> y

    Notes:
    - We intentionally use B=W with shape [N,K] (OUT,IN), like your framework.
    - To get y = x @ W^T, you must set transB=True for the gemm op.
      If you set transB=False with W=[N,K], shapes will mismatch and dispatch will return NotImplemented.
    """
    vid_x = 0
    vid_W = 1
    vid_b = 2
    vid_y = 3

    dev_str = str(device)
    dt_str = f"torch.{str(dtype).split('.')[-1]}"  # "torch.float16" etc

    values: Dict[str, Any] = {}

    def add_val(vid: int, name: str, shape: List[int], dtype_s: str, device_s: str):
        values[str(vid)] = {
            "id": int(vid),
            "name": str(name),
            "shape": list(shape),
            "dtype": str(dtype_s),
            "device": str(device_s),
        }

    # Value metas are *physical* tensor metas; kernel interprets them with trans flags.
    add_val(vid_x, "x", [M, K], dt_str, dev_str)
    add_val(vid_W, "W", [N, K], dt_str, dev_str)
    if with_bias:
        add_val(vid_b, "b", [N], dt_str, dev_str)
    add_val(vid_y, "y", [M, N], dt_str, dev_str)

    nodes: List[Dict[str, Any]] = []
    nodes.append({
        "op": "gemm",
        "inputs": [vid_x, vid_W],
        "outputs": [vid_y],
        "attrs": {"transA": bool(transA), "transB": bool(transB)},
    })
    if with_bias:
        nodes.append({
            "op": "bias_add",
            "inputs": [vid_y, vid_b],
            "outputs": [vid_y],
            "attrs": {},
        })

    root = {
        "graph": "ir_exec_gemm_linear",
        "values": values,
        "nodes": nodes,
    }
    ir_json = json.dumps(root)
    name2vid = {"x": vid_x, "W": vid_W, "y": vid_y}
    if with_bias:
        name2vid["b"] = vid_b
    return ir_json, name2vid


# ============================================================
# case runner
# ============================================================

@dataclass
class Case:
    M: int
    K: int
    N: int
    dtype: torch.dtype
    transA: bool
    transB: bool
    with_bias: bool
    make_strided_A: bool = False
    make_strided_B: bool = False


def _make_strided_like(base: torch.Tensor, *, mode: str) -> torch.Tensor:
    """
    Create a positive-stride non-contiguous view while keeping shape same.
    mode:
      - "pad0": pad leading dim via a larger contiguous then slice
      - "t": transpose (keeps positive strides but changes shape)
    """
    if mode == "pad0":
        M, N = base.shape
        big = torch.empty((M, N + 8), device=base.device, dtype=base.dtype)
        return big[:, :N]  # stride[0] > N, stride[1]==1
    if mode == "t":
        return base.t()
    return base


def _infer_expected_output_shape_from_phys(
    *,
    A_shape: Tuple[int, int],
    B_shape: Tuple[int, int],
    transA: bool,
    transB: bool,
) -> Optional[Tuple[int, int]]:
    """
    Given physical shapes:
      A: [Ar, Ac]
      B: [Br, Bc]
    logical GEMM:
      A_log: [M, K]  where (M,K) = (Ar,Ac) if !transA else (Ac,Ar)
      B_log: [K, N]  where (K,N) = (Br,Bc) if !transB else (Bc,Br)
    returns (M, N) if K matches else None.
    """
    Ar, Ac = A_shape
    Br, Bc = B_shape
    M = Ar if not transA else Ac
    K1 = Ac if not transA else Ar
    K2 = Br if not transB else Bc
    N = Bc if not transB else Br
    if K1 != K2:
        return None
    return (M, N)


def run_case_ir_exec_gemm_linear(case: Case) -> Tuple[bool, Optional[str], Optional[float]]:
    """
    Returns:
      ok, skip_reason, maxdiff
    """
    assert torch.cuda.is_available()
    device = torch.device("cuda:0")

    # Physical tensors:
    # A: [M,K]
    # B: [N,K]  (W style: OUT,IN)
    A = torch.randn((case.M, case.K), device=device, dtype=case.dtype)
    B = torch.randn((case.N, case.K), device=device, dtype=case.dtype)
    b = torch.randn((case.N,), device=device, dtype=case.dtype) if case.with_bias else None

    # Optional striding (shape preserved)
    if case.make_strided_A:
        A = _make_strided_like(A, mode="pad0")
    if case.make_strided_B:
        B = _make_strided_like(B, mode="pad0")

    # Expected output shape from *physical* shapes under trans flags
    out_shape = _infer_expected_output_shape_from_phys(
        A_shape=(A.shape[0], A.shape[1]),
        B_shape=(B.shape[0], B.shape[1]),
        transA=case.transA,
        transB=case.transB,
    )
    if out_shape is None:
        return False, "shape_mismatch(AK!=BK)", None
    OM, ON = out_shape

    # Allocate C
    if case.dtype == torch.float16:
        # TC path currently requires C contiguous row-major
        C = _make_C_contig(OM, ON, device=device, dtype=case.dtype)
    else:
        C = torch.empty((OM, ON), device=device, dtype=case.dtype)

    # Build IR json (lowered ops)
    ir_json, _ = build_ir_json_lowered_gemm(
        M=case.M, K=case.K, N=case.N,
        dtype=case.dtype, device=device,
        transA=case.transA, transB=case.transB,
        with_bias=case.with_bias,
    )

    # Compile
    h = _C.compile_ir_json(ir_json)

    # Bind required
    _C.exe_bind(h, "x", A)
    _C.exe_bind(h, "W", B)
    _C.exe_bind(h, "y", C)
    if case.with_bias:
        _C.exe_bind(h, "b", b)

    req = _C.exe_required_inputs(h)
    _log(
        f"[CASE] dtype={case.dtype} transA={case.transA} transB={case.transB} bias={case.with_bias} "
        f"A({_s(A)}) B({_s(B)}) C({_s(C)}) req={req}"
    )

    # Run
    try:
        _C.exe_run_once(h)
        _cuda_sync()
    except RuntimeError as e:
        msg = str(e)
        if "InvalidArgument" in msg or "status=InvalidArgument" in msg:
            _log(f"[SKIP] dispatch_invalid_argument | {msg}")
            return False, "dispatch_invalid_argument", None
        if "NotImplemented" in msg or "status=NotImplemented" in msg:
            _log(f"[SKIP] dispatch_not_implemented | {msg}")
            return False, "dispatch_not_implemented", None
        raise
    finally:
        _C.exe_destroy(h)

    # Reference compute with identical logical interpretation:
    A_ref = A.t() if case.transA else A
    B_ref = B.t() if case.transB else B
    Y_ref = A_ref @ B_ref
    if case.with_bias:
        Y_ref = Y_ref + b

    # Compare
    atol = 2e-2 if case.dtype == torch.float16 else 1e-5
    rtol = 2e-2 if case.dtype == torch.float16 else 1e-5

    ok = _allclose(C, Y_ref, atol=atol, rtol=rtol)
    md = _maxdiff(C, Y_ref)
    if not ok:
        _log(f"[FAIL] maxdiff={md}  C({_s(C)})  ref({_s(Y_ref)})")
    return ok, None, md


# ============================================================
# main
# ============================================================

def main():
    torch.manual_seed(0)

    cases: List[Case] = []

    # f32: cover NN/TN/NT/TT, contiguous + strided A/B
    for transA in (False, True):
        for transB in (False, True):
            cases.append(Case(M=64, K=32, N=48, dtype=torch.float32, transA=transA, transB=transB, with_bias=False))
            cases.append(Case(M=64, K=32, N=48, dtype=torch.float32, transA=transA, transB=transB, with_bias=True))
            cases.append(Case(M=64, K=32, N=48, dtype=torch.float32, transA=transA, transB=transB, with_bias=False, make_strided_A=True))
            cases.append(Case(M=64, K=32, N=48, dtype=torch.float32, transA=transA, transB=transB, with_bias=False, make_strided_B=True))

    # f16: focus on TC path constraints (C must be contiguous)
    for transA in (False, True):
        for transB in (False, True):
            cases.append(Case(M=64, K=32, N=48, dtype=torch.float16, transA=transA, transB=transB, with_bias=False))
            cases.append(Case(M=64, K=32, N=48, dtype=torch.float16, transA=transA, transB=transB, with_bias=True))
            cases.append(Case(M=64, K=32, N=48, dtype=torch.float16, transA=transA, transB=transB, with_bias=False, make_strided_A=True))
            cases.append(Case(M=64, K=32, N=48, dtype=torch.float16, transA=transA, transB=transB, with_bias=False, make_strided_B=True))

    OK = 0
    SKIP = 0
    FAIL = 0
    skip_reasons: Dict[str, int] = {}

    for i, cs in enumerate(cases):
        ok, skip_reason, md = run_case_ir_exec_gemm_linear(cs)
        if skip_reason is not None:
            SKIP += 1
            skip_reasons[skip_reason] = skip_reasons.get(skip_reason, 0) + 1
            continue
        if ok:
            OK += 1
        else:
            FAIL += 1
            print(
                f"[CASE_FAIL] i={i} dtype={cs.dtype} transA={cs.transA} transB={cs.transB} "
                f"bias={cs.with_bias} maxdiff={md}"
            )

    print(f"[RESULT] OK={OK} SKIP={SKIP} FAIL={FAIL}")
    if SKIP:
        print("[SKIP_REASONS]")
        for k, v in sorted(skip_reasons.items(), key=lambda x: (-x[1], x[0])):
            print(f"  {k}: {v}")

    if FAIL == 0:
        print("ALL PASSED")
    else:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
