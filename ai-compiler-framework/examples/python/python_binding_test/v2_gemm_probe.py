from __future__ import annotations
import sys
from pathlib import Path
import torch
import struct

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from aicf_cuda import _C


def pack_gemm(trans_a: int = 0, trans_b: int = 0) -> bytes:
    return struct.pack("<ii", int(trans_a), int(trans_b))


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def make_A_B(M: int, K: int, N: int, ta: bool, tb: bool, dtype):
    device = torch.device("cuda:0")

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


def torch_gemm_ref(A: torch.Tensor, B: torch.Tensor, ta: bool, tb: bool):
    A2 = A.t() if ta else A
    B2 = B.t() if tb else B
    return A2 @ B2


def run_f32(M, K, N, ta=False, tb=False):
    A, B = make_A_B(M, K, N, ta, tb, torch.float32)

    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()
    C_ref = torch_gemm_ref(A, B, ta, tb)

    _C.op_call(
        int(_C.OpKind.Gemm),
        [A, B],
        [C],
        0,
        pack_gemm(ta, tb),
        0,
    )

    d = maxabs_delta(C, C_ref)
    print(f"[F32] ta={int(ta)} tb={int(tb)} max|delta|={d:.3e}")
    return d


def run_f16_tc(M, K, N, ta=False, tb=False):
    # TC path: your kernel requires C contiguous row-major (we satisfy)
    A, B = make_A_B(M, K, N, ta, tb, torch.float16)

    C = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()
    C_ref = torch_gemm_ref(A.float(), B.float(), ta, tb).half()

    _C.op_call(
        int(_C.OpKind.Gemm),
        [A, B],
        [C],
        0,
        pack_gemm(ta, tb),
        0,
    )

    d = maxabs_delta(C, C_ref)
    print(f"[F16-TC] ta={int(ta)} tb={int(tb)} max|delta|={d:.3e}")
    return d


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    print("Gemm enum value =", int(_C.OpKind.Gemm))

    worst = 0.0
    for ta in (False, True):
        for tb in (False, True):
            worst = max(worst, run_f32(64, 48, 80, ta, tb))
    print("[F32] worst max|delta| =", worst)

    worst_tc = 0.0
    for ta in (False, True):
        for tb in (False, True):
            # WMMA wants multiples of 16 for best coverage; pick 64/64/64
            worst_tc = max(worst_tc, run_f16_tc(64, 64, 64, ta, tb))
    print("[F16-TC] worst max|delta| =", worst_tc)

    # NEG: wrong C shape
    A, B = make_A_B(8, 4, 7, False, False, torch.float32)
    C_bad = torch.empty(8, 8, device="cuda", dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.Gemm), [A, B], [C_bad], 0, pack_gemm(0, 0), 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
