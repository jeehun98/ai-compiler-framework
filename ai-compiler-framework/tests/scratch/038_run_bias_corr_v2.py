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
def ref_biascorr(step_i32_1: torch.Tensor, beta1: float, beta2: float) -> tuple[torch.Tensor, torch.Tensor]:
    # step clamp to >=1 (kernel behavior)
    # step_i32_1: shape=(1,), int32
    t = int(step_i32_1.item())
    t = max(t, 1)
    tt = torch.tensor([float(t)], device=step_i32_1.device, dtype=torch.float32)  # shape=(1,)
    b1 = torch.tensor([beta1], device=step_i32_1.device, dtype=torch.float32)
    b2 = torch.tensor([beta2], device=step_i32_1.device, dtype=torch.float32)
    bc1_inv = 1.0 / (1.0 - torch.pow(b1, tt))
    bc2_inv = 1.0 / (1.0 - torch.pow(b2, tt))
    return bc1_inv, bc2_inv


@torch.no_grad()
def run(step_val: int, beta1: float, beta2: float):
    device = "cuda"

    m = aicf.Model(dtype="i32", device=device)
    s = m.input("step", aicf.TensorSpec((1,), "i32", device))
    bc1, bc2 = m.add(aicf.BiasCorr(name="bc", beta1=beta1, beta2=beta2), s)
    m.output("bc1", bc1)
    m.output("bc2", bc2)

    S = torch.tensor([step_val], device=device, dtype=torch.int32).contiguous()
    ref1, ref2 = ref_biascorr(S, beta1, beta2)

    exe = aicf.CudaExecutor()
    out = exe.run(m, {"step": S})
    O1 = out["bc1"]
    O2 = out["bc2"]

    d1 = maxabs_delta(O1, ref1)
    d2 = maxabs_delta(O2, ref2)
    d = max(d1, d2)

    print(
        f"[rank1] step={step_val} beta1={beta1} beta2={beta2} "
        f"bc1={float(O1[0].item()):.6g} bc2={float(O2[0].item()):.6g} max|d|={d:.3e}"
    )
    return d


def main():
    torch.manual_seed(0)

    worst = 0.0
    for step in (1, 10, 0):
        worst = max(worst, run(step, 0.9, 0.999))
    print("[OK] worst =", worst)

    print("[NEG] wrong schema / dtype tests stay in _C-only probe.")


if __name__ == "__main__":
    main()
