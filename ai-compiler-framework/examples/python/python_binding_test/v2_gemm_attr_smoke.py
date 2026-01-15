# examples/python/python_framework_test/v2_gemm_attr_smoke.py
from __future__ import annotations

import sys
from pathlib import Path
import random
from dataclasses import dataclass

import torch

# --- auto PYTHONPATH bootstrap ---
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]               # repo root
EX_PY = ROOT / "examples" / "python" # contains aicf_fw
BUILD_PY = ROOT / "build" / "python" # contains built extension (aicf_cuda)

for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
# --------------------------------

from aicf_fw.backend import set_backend, get_backend
from aicf_fw.backend.aicf_backend import AICFBackend


def _maxdiff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _assert_close(a: torch.Tensor, b: torch.Tensor, *, atol: float, rtol: float, msg: str):
    if not torch.allclose(a, b, atol=atol, rtol=rtol):
        raise AssertionError(f"{msg} allclose failed: maxdiff={_maxdiff(a,b)} atol={atol} rtol={rtol}")


def _gemm_ref(A: torch.Tensor, B: torch.Tensor, *, transA: bool, transB: bool) -> torch.Tensor:
    a = A.t() if transA else A
    b = B.t() if transB else B
    return a @ b


def _make_stride_view_2d(x: torch.Tensor, mode: str) -> torch.Tensor:
    """
    shape는 유지하면서 stride만 바꿔서 launcher의 논리뷰(transA/transB) 테스트를 깔끔히 한다.
    - contig: 그대로
    - t: transpose view (shape swap됨) -> 이건 "view 테스트"가 아니라 "shape swap"이라 케이스가 꼬임.
         그래서 여기서는 제거.
    - as_strided: 동일 shape, stride만 바꿈 (양수 stride 유지)
    """
    if mode == "contig":
        return x
    if mode == "as_strided":
        # 같은 shape로, row stride를 좀 키워서 "strided" 만들기
        # 주의: storage 충분해야 하므로 더 큰 텐서에서 view 뽑는 방식으로 안전하게.
        M, N = x.shape
        big = torch.empty((M, N + 8), device=x.device, dtype=x.dtype)
        base = big[:, :N]  # shape (M,N), stride[0]=N+8, stride[1]=1 (양수)
        base.copy_(x)
        return base
    raise ValueError(mode)


def _alloc_C(M: int, N: int, *, dtype: torch.dtype, device: torch.device, require_contig: bool) -> torch.Tensor:
    if require_contig:
        return torch.empty((M, N), device=device, dtype=dtype)
    # f32은 C도 strided로 한번 더 가능
    big = torch.empty((M, N + 4), device=device, dtype=dtype)
    return big[:, :N]


@dataclass
class Case:
    M: int
    K: int
    N: int
    transA: bool
    transB: bool
    dtype: torch.dtype
    viewA: str
    viewB: str
    attr_mode: str  # "bool" or "int"


def run_one(bk, c: Case, device: torch.device) -> str:
    # IMPORTANT:
    # transA/transB는 "launcher의 논리 view"로 처리해야 하므로
    # A0,B0는 항상 "물리적 shape"를 (M,K),(K,N)로 유지한다.
    A0 = torch.randn((c.M, c.K), device=device, dtype=c.dtype)
    B0 = torch.randn((c.K, c.N), device=device, dtype=c.dtype)

    A = _make_stride_view_2d(A0, c.viewA)
    B = _make_stride_view_2d(B0, c.viewB)

    # logical shape after transpose flags
    A_log = A.t() if c.transA else A
    B_log = B.t() if c.transB else B
    if A_log.shape[1] != B_log.shape[0]:
        return "SKIP(shape)"

    M = A_log.shape[0]
    N = B_log.shape[1]

    require_contig_C = (c.dtype == torch.float16)  # 네 TC variant 조건
    C = _alloc_C(M, N, dtype=c.dtype, device=device, require_contig=require_contig_C)

    ref = _gemm_ref(A, B, transA=c.transA, transB=c.transB).to(c.dtype)

    if c.attr_mode == "bool":
        attrs = {"transA": bool(c.transA), "transB": bool(c.transB)}
    else:
        # 브리지 bool-drop 의심용: int로 강제
        attrs = {"transA": int(c.transA), "transB": int(c.transB)}

    bk.op_call_out("gemm", [A, B], [C], attrs)

    if c.dtype == torch.float32:
        _assert_close(C, ref, atol=1e-4, rtol=1e-4,
                      msg=f"f32 MKN={c.M},{c.K},{c.N} transA={c.transA} transB={c.transB} viewA={c.viewA} viewB={c.viewB} attr={c.attr_mode}")
    else:
        # half output: 널널하게
        _assert_close(C, ref, atol=3e-2, rtol=3e-2,
                      msg=f"f16 MKN={c.M},{c.K},{c.N} transA={c.transA} transB={c.transB} viewA={c.viewA} viewB={c.viewB} attr={c.attr_mode}")

    return "OK"


def main():
    torch.manual_seed(0)
    random.seed(0)

    device = torch.device("cuda", 0)

    set_backend(AICFBackend())
    bk = get_backend()

    sizes = [(7, 9, 11), (16, 16, 16), (33, 17, 29)]
    trans_flags = [(False, False), (False, True), (True, False), (True, True)]
    views = ["contig", "as_strided"]
    dtypes = [torch.float32, torch.float16]
    attr_modes = ["bool", "int"]

    ok = skip = 0
    for (M, K, N) in sizes:
        for (ta, tb) in trans_flags:
            for dt in dtypes:
                for vA in views:
                    for vB in views:
                        for am in attr_modes:
                            c = Case(M, K, N, ta, tb, dt, vA, vB, am)
                            r = run_one(bk, c, device)
                            if r.startswith("SKIP"):
                                skip += 1
                            else:
                                ok += 1

    print(f"[RESULT] OK={ok} SKIP={skip}")
    print("ALL PASSED")


if __name__ == "__main__":
    main()
