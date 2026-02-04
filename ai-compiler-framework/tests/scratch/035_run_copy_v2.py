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
def run(dtype: str, shape):
    device = "cuda"
    m = aicf.Model(dtype=dtype, device=device)

    x = m.input("x", aicf.TensorSpec(shape, dtype, device))
    y = m.add(aicf.Copy(name="cpy"), x)
    m.output("y", y)

    torch_dtype = torch.float16 if dtype == "f16" else torch.float32
    X = torch.randn(*shape, device=device, dtype=torch_dtype).contiguous()

    exe = aicf.CudaExecutor()
    Y = exe.run(m, {"x": X})["y"]

    if torch_dtype == torch.float16:
        d = maxabs_delta(Y.float(), X.float())
    else:
        d = maxabs_delta(Y, X)

    print(f"[{dtype}] shape={tuple(shape)} max|delta|={d:.3e}")
    return d


def main():
    torch.manual_seed(0)

    worst32 = 0.0
    for shape in [(1024,), (64, 256), (8, 32, 128)]:
        worst32 = max(worst32, run("f32", shape))
    print("[f32] worst =", worst32)

    worst16 = 0.0
    for shape in [(1024,), (64, 256), (7, 33, 127)]:
        worst16 = max(worst16, run("f16", shape))
    print("[f16] worst =", worst16)

    print("[NEG] keep dtype/shape negative tests in _C-only harness for Copy")


if __name__ == "__main__":
    main()
