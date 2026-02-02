from __future__ import annotations

import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]  # python/aicf_v2
SRC = ROOT / "src"
sp = str(SRC)
if sp not in sys.path:
    sys.path.insert(0, sp)

import aicf_v2 as aicf


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


@torch.no_grad()
def run():
    device = "cuda"
    m = aicf.Model(dtype="i32", device=device)

    # v2 contract: step is rank-1 scalar (1,)
    s = m.input("S", aicf.TensorSpec((1,), "i32", device))
    so = m.add(aicf.StepInc(name="step"), s)
    m.output("SO", so)

    S = torch.zeros((1,), device=device, dtype=torch.int32).contiguous()
    ref = S + 1

    exe = aicf.CudaExecutor()
    out = exe.run(m, {"S": S})["SO"]

    d = maxabs_delta(out, ref)
    print(f"[oop] shape={(1,)} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)

    worst = run()
    print("[OK] worst =", worst)
    print("[note] StepInc is an optimizer step-counter op; only (1,) i32 is supported in v2.")


if __name__ == "__main__":
    main()
