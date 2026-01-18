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


def fourcc(s: str) -> int:
    assert len(s) == 4
    return int.from_bytes(s.encode("ascii"), "little")


SCHEMA_BADD = fourcc("BADD")


def pack_bias_add(axis: int = -1) -> bytes:
    # BiasAddAttrV0: int64 axis
    return struct.pack("<q", int(axis))


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def torch_bias_add_ref(Y: torch.Tensor, bias: torch.Tensor, axis: int = -1) -> torch.Tensor:
    # Only last-dim supported in our op. axis can be -1 or last dim.
    # Broadcast bias over all leading dims.
    r = Y.dim()
    if axis == -1:
        axis = r - 1
    shape = [1] * r
    shape[axis] = bias.numel()
    return Y + bias.view(*shape)


def run_f32(shape, axis=-1):
    device = torch.device("cuda:0")
    Y = torch.randn(*shape, device=device, dtype=torch.float32).contiguous()
    N = shape[-1]
    bias = torch.randn(N, device=device, dtype=torch.float32).contiguous()
    O = torch.empty_like(Y).contiguous()
    ref = torch_bias_add_ref(Y, bias, axis)

    _C.op_call(
        int(_C.OpKind.BiasAdd),
        [Y, bias],
        [O],
        SCHEMA_BADD,
        pack_bias_add(axis),
        0,
    )

    d = maxabs_delta(O, ref)
    print(f"[F32] shape={tuple(shape)} axis={axis} max|delta|={d:.3e}")
    return d


def run_f16(shape, axis=-1):
    device = torch.device("cuda:0")
    Y = torch.randn(*shape, device=device, dtype=torch.float16).contiguous()
    N = shape[-1]
    bias = torch.randn(N, device=device, dtype=torch.float16).contiguous()
    O = torch.empty_like(Y).contiguous()
    # ref in fp32 then cast for better numerical stability comparison
    ref = torch_bias_add_ref(Y.float(), bias.float(), axis).half()

    _C.op_call(
        int(_C.OpKind.BiasAdd),
        [Y, bias],
        [O],
        SCHEMA_BADD,
        pack_bias_add(axis),
        0,
    )

    d = maxabs_delta(O, ref)
    print(f"[F16] shape={tuple(shape)} axis={axis} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)

    print("BiasAdd enum value =", int(_C.OpKind.BiasAdd))
    print("schema_id(BADD) =", hex(SCHEMA_BADD))

    # -----------------
    # Positive cases
    # -----------------
    worst_f32 = 0.0
    worst_f32 = max(worst_f32, run_f32((64, 256), axis=-1))
    worst_f32 = max(worst_f32, run_f32((8, 32, 128), axis=-1))
    print("[F32] worst max|delta| =", worst_f32)

    worst_f16 = 0.0
    worst_f16 = max(worst_f16, run_f16((64, 256), axis=-1))      # N even => can hit half2
    worst_f16 = max(worst_f16, run_f16((7, 33, 127), axis=-1))   # N odd  => forces naive half
    print("[F16] worst max|delta| =", worst_f16)

    # -----------------
    # Negative cases
    # -----------------

    # NEG1: axis not last-dim (should error)
    Y = torch.randn(4, 5, 6, device="cuda", dtype=torch.float32).contiguous()
    bias = torch.randn(5, device="cuda", dtype=torch.float32).contiguous()
    O = torch.empty_like(Y).contiguous()

    try:
        _C.op_call(
            int(_C.OpKind.BiasAdd),
            [Y, bias],
            [O],
            SCHEMA_BADD,
            pack_bias_add(axis=1),  # not supported
            0,
        )
        print("[NEG axis] unexpected OK")
    except RuntimeError as e:
        print("[NEG axis] ok:", str(e).splitlines()[0])

    # NEG2: bias length mismatch (should error)
    Y = torch.randn(4, 7, device="cuda", dtype=torch.float32).contiguous()
    bias_bad = torch.randn(8, device="cuda", dtype=torch.float32).contiguous()  # should be 7
    O = torch.empty_like(Y).contiguous()
    try:
        _C.op_call(
            int(_C.OpKind.BiasAdd),
            [Y, bias_bad],
            [O],
            SCHEMA_BADD,
            pack_bias_add(axis=-1),
            0,
        )
        print("[NEG bias shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG bias shape] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
