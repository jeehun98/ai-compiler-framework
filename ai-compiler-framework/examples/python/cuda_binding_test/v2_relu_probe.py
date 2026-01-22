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


def relu_ref(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp_min(x, 0)


def run_f32(shape):
    x = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    o = torch.empty_like(x).contiguous()
    ref = relu_ref(x)

    _C.op_call(
        int(_C.OpKind.EltwiseRelu),
        [x],
        [o],
        0,
        b"",
        0,
    )

    d = maxabs_delta(o, ref)
    print(f"[F32] shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def run_f16(shape):
    x = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    o = torch.empty_like(x).contiguous()
    # reference in f16 (relu is exact in half for these ops)
    ref = relu_ref(x)

    _C.op_call(
        int(_C.OpKind.EltwiseRelu),
        [x],
        [o],
        0,
        b"",
        0,
    )

    d = maxabs_delta(o, ref)
    print(f"[F16] shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)

    print("EltwiseRelu enum value =", int(_C.OpKind.EltwiseRelu))

    # -----------------
    # Positive cases
    # -----------------
    worst_f32 = 0.0
    worst_f32 = max(worst_f32, run_f32((1024,)))        # 1D
    worst_f32 = max(worst_f32, run_f32((64, 256)))      # 2D
    worst_f32 = max(worst_f32, run_f32((8, 32, 128)))   # 3D
    print("[F32] worst max|delta| =", worst_f32)

    worst_f16 = 0.0
    worst_f16 = max(worst_f16, run_f16((1024,)))        # 1D
    worst_f16 = max(worst_f16, run_f16((64, 256)))      # 2D (even numel -> can hit half2)
    worst_f16 = max(worst_f16, run_f16((7, 33, 127)))   # 3D (odd numel -> forces naive half)
    print("[F16] worst max|delta| =", worst_f16)

    # -----------------
    # Negative cases
    # -----------------
    # NEG1: dtype mismatch (input f32, output f16)
    x = torch.randn(8, 7, device="cuda", dtype=torch.float32).contiguous()
    o_bad = torch.empty(8, 7, device="cuda", dtype=torch.float16).contiguous()

    try:
        _C.op_call(int(_C.OpKind.EltwiseRelu), [x], [o_bad], 0, b"", 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])

    # NEG2: shape mismatch
    x = torch.randn(8, 7, device="cuda", dtype=torch.float32).contiguous()
    o_bad = torch.empty(8, 8, device="cuda", dtype=torch.float32).contiguous()

    try:
        _C.op_call(int(_C.OpKind.EltwiseRelu), [x], [o_bad], 0, b"", 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
