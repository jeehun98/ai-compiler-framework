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


def relu_bwd_ref(Y: torch.Tensor, dOut: torch.Tensor) -> torch.Tensor:
    return torch.where(Y > 0, dOut, torch.zeros_like(dOut))


def run_f32(shape):
    # forward output Y (not pre-activation X)
    X = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    Y = torch.clamp_min(X, 0).contiguous()
    dOut = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    dY = torch.empty_like(Y).contiguous()

    ref = relu_bwd_ref(Y, dOut)

    _C.op_call(
        int(_C.OpKind.ReluBwd),
        [dOut, Y],      # contract: inputs[0]=dOut, inputs[1]=Y
        [dY],
        0,
        b"",
        0,
    )

    d = maxabs_delta(dY, ref)
    print(f"[F32] shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def run_f16(shape):
    X = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    Y = torch.clamp_min(X, 0).contiguous()
    dOut = torch.randn(*shape, device="cuda", dtype=torch.float16).contiguous()
    dY = torch.empty_like(Y).contiguous()

    ref = relu_bwd_ref(Y, dOut)

    _C.op_call(
        int(_C.OpKind.ReluBwd),
        [dOut, Y],
        [dY],
        0,
        b"",
        0,
    )

    d = maxabs_delta(dY, ref)
    print(f"[F16] shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)

    print("EltwiseReluBwd enum value =", int(_C.OpKind.ReluBwd))

    worst32 = 0.0
    worst32 = max(worst32, run_f32((1024,)))
    worst32 = max(worst32, run_f32((64, 256)))
    worst32 = max(worst32, run_f32((8, 32, 128)))
    print("[F32] worst max|delta| =", worst32)

    worst16 = 0.0
    worst16 = max(worst16, run_f16((1024,)))       # even numel -> likely half2
    worst16 = max(worst16, run_f16((64, 256)))     # even numel -> likely half2
    worst16 = max(worst16, run_f16((7, 33, 127)))  # odd numel -> forces naive
    print("[F16] worst max|delta| =", worst16)

    # NEG: dtype mismatch (Y f32, dOut f16)
    Y = torch.zeros(8, 7, device="cuda", dtype=torch.float32).contiguous()
    dOut = torch.zeros(8, 7, device="cuda", dtype=torch.float16).contiguous()
    dY = torch.empty_like(Y).contiguous()
    try:
        _C.op_call(int(_C.OpKind.ReluBwd), [dOut, Y], [dY], 0, b"", 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])

    # NEG: shape mismatch
    Y = torch.zeros(8, 7, device="cuda", dtype=torch.float32).contiguous()
    dOut = torch.zeros(8, 8, device="cuda", dtype=torch.float32).contiguous()
    dY = torch.empty_like(Y).contiguous()
    try:
        _C.op_call(int(_C.OpKind.ReluBwd), [dOut, Y], [dY], 0, b"", 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
