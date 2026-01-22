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


def pack_sgds(lr: float) -> bytes:
    # schema 'SGDS' payload: float32 lr (little-endian)
    return struct.pack("<f", float(lr))


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def ref_sgd_step(P: torch.Tensor, G: torch.Tensor, lr: float) -> torch.Tensor:
    return P - lr * G


def run_f32(shape, lr: float, inplace: bool = False) -> float:
    device = torch.device("cuda:0")
    P = torch.randn(*shape, device=device, dtype=torch.float32).contiguous()
    G = torch.randn(*shape, device=device, dtype=torch.float32).contiguous()

    if inplace:
        O = P  # in-place allowed (O aliases P)
    else:
        O = torch.empty_like(P).contiguous()

    O_ref = ref_sgd_step(P, G, lr)

    _C.op_call(
        int(_C.OpKind.SgdStep),
        [P, G],
        [O],
        0x53444753,  # 'SGDS'
        pack_sgds(lr),
        0,
    )

    d = maxabs_delta(O, O_ref)
    tag = "inplace" if inplace else "oop"
    print(f"[F32 {tag}] shape={tuple(shape)} lr={lr:g} max|delta|={d:.3e}")
    return d


def run_f16(shape, lr: float, inplace: bool = False) -> float:
    device = torch.device("cuda:0")
    P = torch.randn(*shape, device=device, dtype=torch.float16).contiguous()
    G = torch.randn(*shape, device=device, dtype=torch.float16).contiguous()

    if inplace:
        O = P
    else:
        O = torch.empty_like(P).contiguous()

    # reference: do math in fp32 then cast back (matches kernel intent best)
    O_ref = ref_sgd_step(P.float(), G.float(), lr).half()

    _C.op_call(
        int(_C.OpKind.SgdStep),
        [P, G],
        [O],
        0x53444753,  # 'SGDS'
        pack_sgds(lr),
        0,
    )

    d = maxabs_delta(O, O_ref)
    tag = "inplace" if inplace else "oop"
    print(f"[F16 {tag}] shape={tuple(shape)} lr={lr:g} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)
    print("SgdStep enum value =", int(_C.OpKind.SgdStep))
    print("schema_id(SGDS) =", hex(0x53444753))

    lr = 1e-3

    # Positive: f32
    worst_f32 = 0.0
    worst_f32 = max(worst_f32, run_f32((1024,), lr, inplace=False))
    worst_f32 = max(worst_f32, run_f32((64, 256), lr, inplace=False))
    worst_f32 = max(worst_f32, run_f32((8, 32, 128), lr, inplace=False))
    worst_f32 = max(worst_f32, run_f32((64, 256), lr, inplace=True))
    print("[F32] worst max|delta| =", worst_f32)

    # Positive: f16
    worst_f16 = 0.0
    worst_f16 = max(worst_f16, run_f16((1024,), lr, inplace=False))
    worst_f16 = max(worst_f16, run_f16((64, 256), lr, inplace=False))
    worst_f16 = max(worst_f16, run_f16((7, 33, 127), lr, inplace=False))  # odd numel -> scalar path
    worst_f16 = max(worst_f16, run_f16((64, 256), lr, inplace=True))
    print("[F16] worst max|delta| =", worst_f16)

    # NEG: wrong dtype (mix)
    device = torch.device("cuda:0")
    P = torch.randn(128, device=device, dtype=torch.float32).contiguous()
    G = torch.randn(128, device=device, dtype=torch.float16).contiguous()
    O = torch.empty_like(P).contiguous()
    try:
        _C.op_call(int(_C.OpKind.SgdStep), [P, G], [O], 0x53444753, pack_sgds(lr), 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])

    # NEG: shape mismatch
    P = torch.randn(128, device=device, dtype=torch.float32).contiguous()
    G = torch.randn(127, device=device, dtype=torch.float32).contiguous()
    O = torch.empty_like(P).contiguous()
    try:
        _C.op_call(int(_C.OpKind.SgdStep), [P, G], [O], 0x53444753, pack_sgds(lr), 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])

    # NEG: forbid out aliases grad (unsafe)
    P = torch.randn(256, device=device, dtype=torch.float32).contiguous()
    G = torch.randn(256, device=device, dtype=torch.float32).contiguous()
    O_alias_grad = G  # should be rejected
    try:
        _C.op_call(int(_C.OpKind.SgdStep), [P, G], [O_alias_grad], 0x53444753, pack_sgds(lr), 0)
        print("[NEG alias grad] unexpected OK")
    except RuntimeError as e:
        print("[NEG alias grad] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
