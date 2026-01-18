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


def run(shape, inplace: bool):
    device = torch.device("cuda:0")
    S = torch.zeros(*shape, device=device, dtype=torch.int32).contiguous()
    S_ref = S.clone()

    # ref: +1
    S_ref += 1

    if inplace:
        SO = S
    else:
        SO = S.clone()

    _C.op_call(
        int(_C.OpKind.StepInc),
        [S],
        [SO],
        0,          # schema_id none
        b"",        # attr bytes empty
        0,
    )

    d = maxabs_delta(SO, S_ref)
    tag = "inplace" if inplace else "oop"
    print(f"[{tag}] shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def run_scalar0d(inplace: bool):
    device = torch.device("cuda:0")
    # 0-d scalar
    S = torch.zeros((), device=device, dtype=torch.int32)
    S_ref = S + 1

    if inplace:
        SO = S
    else:
        SO = S.clone()

    _C.op_call(int(_C.OpKind.StepInc), [S], [SO], 0, b"", 0)

    d = maxabs_delta(SO, S_ref)
    tag = "scalar0d_inplace" if inplace else "scalar0d_oop"
    print(f"[{tag}] value={int(SO.item())} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)

    print("StepInc enum value =", int(_C.OpKind.StepInc))

    worst = 0.0
    for shape in [(1,), (1024,), (64, 256)]:
        worst = max(worst, run(shape, inplace=False))
        worst = max(worst, run(shape, inplace=True))

    worst = max(worst, run_scalar0d(inplace=False))
    worst = max(worst, run_scalar0d(inplace=True))

    print("[OK] worst max|delta| =", worst)

    # NEG: dtype mismatch
    S = torch.zeros(128, device="cuda", dtype=torch.float32).contiguous()
    SO = torch.zeros_like(S)
    try:
        _C.op_call(int(_C.OpKind.StepInc), [S], [SO], 0, b"", 0)
        print("[NEG dtype] unexpected OK")
    except RuntimeError as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])

    # NEG: shape mismatch
    S = torch.zeros(16, device="cuda", dtype=torch.int32).contiguous()
    SO = torch.zeros(32, device="cuda", dtype=torch.int32).contiguous()
    try:
        _C.op_call(int(_C.OpKind.StepInc), [S], [SO], 0, b"", 0)
        print("[NEG shape] unexpected OK")
    except RuntimeError as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])


if __name__ == "__main__":
    main()
