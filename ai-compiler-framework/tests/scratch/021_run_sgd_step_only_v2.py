from __future__ import annotations
import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]
SRC = ROOT / "src"
sp = str(SRC)
if sp not in sys.path:
    sys.path.insert(0, sp)

import aicf_v2 as aicf


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def ref_sgd_step(P: torch.Tensor, G: torch.Tensor, lr: float) -> torch.Tensor:
    return P - lr * G


def main():
    torch.manual_seed(0)
    device = "cuda"
    lr = 1e-3

    shape = (64, 256)
    dtype = "f16"  # "f32"도 가능

    m = aicf.Model(dtype=dtype, device=device)

    spec = lambda sh: aicf.TensorSpec(shape=sh, dtype=dtype, device=device)
    P = m.input("P", spec(shape))
    G = m.input("G", spec(shape))

    O = m.add(aicf.SgdStep(name="sgd", lr=lr), P, G)
    m.output("O", O)

    Pt = torch.randn(*shape, device=device, dtype=torch.float16 if dtype == "f16" else torch.float32).contiguous()
    Gt = torch.randn(*shape, device=device, dtype=Pt.dtype).contiguous()

    if dtype == "f16":
        Oref = ref_sgd_step(Pt.float(), Gt.float(), lr).half()
    else:
        Oref = ref_sgd_step(Pt, Gt, lr)

    exe = aicf.CudaExecutor()
    out = exe.run(m, {"P": Pt, "G": Gt})

    print("[OUT KEYS]", list(out.keys()))
    d = maxabs_delta(out["O"], Oref)
    print("max|delta| =", d)
    print("[OK]" if d == 0.0 else "[WARN] nonzero delta")


if __name__ == "__main__":
    main()
