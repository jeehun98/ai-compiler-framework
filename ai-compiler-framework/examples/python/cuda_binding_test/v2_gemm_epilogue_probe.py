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


# launcher.cu 기준:
#   schema_id==0: default transA=0, transB=0, relu=1 (Bias+ReLU)
#   schema_id=='GPEL': <iii> (transA, transB, relu)
def pack_gemm_epilogue(trans_a: int = 0, trans_b: int = 0, relu: int = 1) -> bytes:
    return struct.pack("<iii", int(trans_a), int(trans_b), int(relu))


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


def torch_gemm_epilogue_ref(A: torch.Tensor, B: torch.Tensor, bias: torch.Tensor, ta: bool, tb: bool, relu: bool):
    A2 = A.t() if ta else A
    B2 = B.t() if tb else B
    Y = A2 @ B2
    Y = Y + bias  # bias: (N,)
    if relu:
        Y = torch.relu(Y)
    return Y


def run_f32(M, K, N, ta=False, tb=False, relu=True):
    A, B = make_A_B(M, K, N, ta, tb, torch.float32)

    # bias: (N,)
    bias = torch.randn(N, device="cuda", dtype=torch.float32).contiguous()

    C = torch.empty(M, N, device="cuda", dtype=torch.float32).contiguous()
    C_ref = torch_gemm_epilogue_ref(A, B, bias, ta, tb, relu)

    _C.op_call(
        int(_C.OpKind.GemmEpilogue),
        [A, B, bias],
        [C],
        0,
        pack_gemm_epilogue(ta, tb, int(relu)),
        0,
    )

    d = maxabs_delta(C, C_ref)
    print(f"[F32] ta={int(ta)} tb={int(tb)} relu={int(relu)} max|delta|={d:.3e}")
    return d


def run_f16_tc(M, K, N, ta=False, tb=False, relu=True):
    # TC path: kernel requires C contiguous row-major (we satisfy)
    A, B = make_A_B(M, K, N, ta, tb, torch.float16)

    # NOTE(v0): launcher.cu 구현이 Bias dtype을 f16로 체크하는 버전이라면 bias도 f16로 맞춰야 함.
    # 만약 네가 bias를 f32로 바꿔 구현했다면 여기 dtype만 float32로 바꿔도 됨.
    bias = torch.randn(N, device="cuda", dtype=torch.float16).contiguous()

    C = torch.empty(M, N, device="cuda", dtype=torch.float16).contiguous()

    # ref는 float로 계산 후 half로 (relu까지 동일)
    C_ref = torch_gemm_epilogue_ref(A.float(), B.float(), bias.float(), ta, tb, relu).half()

    _C.op_call(
        int(_C.OpKind.GemmEpilogue),
        [A, B, bias],
        [C],
        0,
        pack_gemm_epilogue(ta, tb, int(relu)),
        0,
    )

    d = maxabs_delta(C, C_ref)
    print(f"[F16-TC] ta={int(ta)} tb={int(tb)} relu={int(relu)} max|delta|={d:.3e}")
    return d


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(0)

    print("GemmEpilogue enum value =", int(_C.OpKind.GemmEpilogue))

    # correctness sweep (f32)
    worst = 0.0
    for relu in (True, False):
        for ta in (False, True):
            for tb in (False, True):
                worst = max(worst, run_f32(64, 48, 80, ta, tb, relu))
    print("[F32] worst max|delta| =", worst)

    # TC path (wmma) wants multiples of 16 for best coverage
    worst_tc = 0.0
    for relu in (True, False):
        for ta in (False, True):
            for tb in (False, True):
                worst_tc = max(worst_tc, run_f16_tc(64, 64, 64, ta, tb, relu))
    print("[F16-TC] worst max|delta| =", worst_tc)

    # NEG1: wrong C shape
    A, B = make_A_B(8, 4, 7, False, False, torch.float32)
    bias = torch.randn(7, device="cuda", dtype=torch.float32).contiguous()
    C_bad = torch.empty(8, 8, device="cuda", dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.GemmEpilogue), [A, B, bias], [C_bad], 0, pack_gemm_epilogue(0, 0, 1), 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])

    # NEG2: wrong bias length
    C = torch.empty(8, 7, device="cuda", dtype=torch.float32).contiguous()
    bias_bad = torch.randn(8, device="cuda", dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.GemmEpilogue), [A, B, bias_bad], [C], 0, pack_gemm_epilogue(0, 0, 1), 0)
        print("[NEG bias] unexpected OK")
    except RuntimeError as e:
        print("[NEG bias] ok:", str(e).splitlines()[0])

    # NEG3: wrong dtype mix (f16 A,B but f32 C) - should fail supported()
    A16, B16 = make_A_B(16, 16, 16, False, False, torch.float16)
    bias16 = torch.randn(16, device="cuda", dtype=torch.float16).contiguous()
    C32 = torch.empty(16, 16, device="cuda", dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.GemmEpilogue), [A16, B16, bias16], [C32], 0, pack_gemm_epilogue(0, 0, 1), 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
