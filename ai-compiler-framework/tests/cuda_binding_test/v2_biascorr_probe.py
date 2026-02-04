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


def pack_bcorr(beta1: float, beta2: float) -> bytes:
    return struct.pack("<ff", float(beta1), float(beta2))


def schema_BCOR() -> int:
    # 'BCOR' little-endian -> 0x524F4342
    return int.from_bytes(b"BCOR", "little")


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def ref_biascorr_f32(step: int, beta1: float, beta2: float, device="cuda:0"):
    t = max(int(step), 1)
    b1 = torch.tensor(beta1, device=device, dtype=torch.float32)
    b2 = torch.tensor(beta2, device=device, dtype=torch.float32)
    tt = torch.tensor(float(t), device=device, dtype=torch.float32)

    bc1_inv = 1.0 / (1.0 - torch.pow(b1, tt))
    bc2_inv = 1.0 / (1.0 - torch.pow(b2, tt))
    return float(bc1_inv.item()), float(bc2_inv.item())


def run(step_val: int, beta1: float, beta2: float, use_rank1_scalar: bool):
    device = torch.device("cuda:0")

    if use_rank1_scalar:
        S = torch.tensor([step_val], device=device, dtype=torch.int32).contiguous()
        O1 = torch.empty((1,), device=device, dtype=torch.float32).contiguous()
        O2 = torch.empty((1,), device=device, dtype=torch.float32).contiguous()
    else:
        S = torch.tensor(step_val, device=device, dtype=torch.int32)  # 0-d
        O1 = torch.empty((), device=device, dtype=torch.float32)
        O2 = torch.empty((), device=device, dtype=torch.float32)

    sid = schema_BCOR()
    attrs = pack_bcorr(beta1, beta2)

    _C.op_call(
        int(_C.OpKind.BiasCorr),
        [S],
        [O1, O2],
        sid,
        attrs,
        0,
    )

    bc1_ref, bc2_ref = ref_biascorr_f32(step_val, beta1, beta2, device=str(device))
    r1 = torch.tensor(bc1_ref, device=device, dtype=torch.float32)
    r2 = torch.tensor(bc2_ref, device=device, dtype=torch.float32)
    d1 = maxabs_delta(O1.float(), r1)
    d2 = maxabs_delta(O2.float(), r2)

    tag = "rank1" if use_rank1_scalar else "rank0"
    print(f"[{tag}] step={step_val} beta1={beta1} beta2={beta2} "
          f"bc1={float(O1.item() if O1.ndim==0 else O1[0].item()):.6g} "
          f"bc2={float(O2.item() if O2.ndim==0 else O2[0].item()):.6g} "
          f"max|d|={max(d1,d2):.3e}")
    return max(d1, d2)


def main():
    torch.manual_seed(0)

    print("BiasCorr enum value =", int(_C.OpKind.BiasCorr))
    print("schema_id(BCOR) =", hex(schema_BCOR()))

    worst = 0.0
    for use_rank1 in (False, True):
        worst = max(worst, run(1, 0.9, 0.999, use_rank1))
        worst = max(worst, run(10, 0.9, 0.999, use_rank1))
        worst = max(worst, run(0, 0.9, 0.999, use_rank1))   # clamp to 1
    print("[OK] worst max|delta| =", worst)

    # NEG: wrong schema id
    device = torch.device("cuda:0")
    S = torch.tensor([5], device=device, dtype=torch.int32).contiguous()
    O1 = torch.empty((1,), device=device, dtype=torch.float32).contiguous()
    O2 = torch.empty((1,), device=device, dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.BiasCorr), [S], [O1, O2], 0x12345678, pack_bcorr(0.9, 0.999), 0)
        print("[NEG schema] unexpected OK")
    except RuntimeError as e:
        print("[NEG schema] ok:", str(e).splitlines()[0])

    # NEG: dtype mismatch (step not int32)
    S_bad = torch.tensor([5], device=device, dtype=torch.float32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.BiasCorr), [S_bad], [O1, O2], schema_BCOR(), pack_bcorr(0.9, 0.999), 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
