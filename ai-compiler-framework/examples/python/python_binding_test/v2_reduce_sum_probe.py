from __future__ import annotations
import sys, struct
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

from aicf_cuda import _C

RSUM = 0x5253554D  # 'RSUM' (ReduceSumAttrV0 schema_id)
def pack_axis(axis: int) -> bytes:
    return struct.pack("<q", int(axis))  # int64 little-endian

def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

def run_f32():
    device = torch.device("cuda:0")
    torch.manual_seed(0)

    M, N = 128, 64
    dY = torch.randn(M, N, device=device, dtype=torch.float32).contiguous()
    dB = torch.empty((N,), device=device, dtype=torch.float32).contiguous()

    ref = dY.sum(dim=0)

    # ✅ 명시 스키마 + axis=0
    _C.op_call(int(_C.OpKind.ReduceSum), [dY], [dB], RSUM, pack_axis(0), 0)

    print("[F32] max|delta| =", maxabs_delta(dB, ref))

def run_f16_to_f32():
    device = torch.device("cuda:0")
    torch.manual_seed(1)

    M, N = 257, 64  # N even이면 half2 fastpath 가능(align에 따라)
    dY = torch.randn(M, N, device=device, dtype=torch.float16).contiguous()
    dB = torch.empty((N,), device=device, dtype=torch.float32).contiguous()

    ref = dY.float().sum(dim=0)

    # ✅ 명시 스키마 + axis=0
    _C.op_call(int(_C.OpKind.ReduceSum), [dY], [dB], RSUM, pack_axis(0), 0)

    print("[F16->F32] max|delta| =", maxabs_delta(dB, ref))

def run_negative_axis1():
    device = torch.device("cuda:0")
    torch.manual_seed(2)

    M, N = 32, 16
    dY = torch.randn(M, N, device=device, dtype=torch.float32).contiguous()
    dB = torch.empty((N,), device=device, dtype=torch.float32).contiguous()

    try:
        _C.op_call(int(_C.OpKind.ReduceSum), [dY], [dB], RSUM, pack_axis(1), 0)
        print("[NEG axis=1] ERROR: expected failure but succeeded")
    except Exception as e:
        print("[NEG axis=1] ok:", str(e)[:200])

def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("ReduceSum enum value =", int(_C.OpKind.ReduceSum))
    run_f32()
    run_f16_to_f32()
    run_negative_axis1()

if __name__ == "__main__":
    main()
