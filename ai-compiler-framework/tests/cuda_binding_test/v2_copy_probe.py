from __future__ import annotations
import sys
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


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def run_f32(shape):
    x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    y = torch.empty_like(x).contiguous()

    _C.op_call(int(_C.OpKind.Copy), [x], [y], 0, b"", 0)

    d = maxabs_delta(y, x)
    print(f"[F32] shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def run_f16(shape):
    x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    y = torch.empty_like(x).contiguous()

    _C.op_call(int(_C.OpKind.Copy), [x], [y], 0, b"", 0)

    d = maxabs_delta(y.float(), x.float())
    print(f"[F16] shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)
    print("Copy enum value =", int(_C.OpKind.Copy))

    worst_f32 = 0.0
    for shape in [(1024,), (64, 256), (8, 32, 128)]:
        worst_f32 = max(worst_f32, run_f32(shape))
    print("[F32] worst max|delta| =", worst_f32)

    worst_f16 = 0.0
    for shape in [(1024,), (64, 256), (7, 33, 127)]:
        worst_f16 = max(worst_f16, run_f16(shape))
    print("[F16] worst max|delta| =", worst_f16)

    # NEG: wrong dtype
    x = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    y = torch.empty(128, device="cuda", dtype=torch.float16).contiguous()
    try:
        _C.op_call(int(_C.OpKind.Copy), [x], [y], 0, b"", 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])

    # NEG: wrong shape
    x = torch.randn(128, device="cuda", dtype=torch.float32).contiguous()
    y = torch.empty(256, device="cuda", dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.Copy), [x], [y], 0, b"", 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
