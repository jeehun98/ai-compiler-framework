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


def ref_mse_grad(pred: torch.Tensor, target: torch.Tensor, scale: float) -> torch.Tensor:
    return (pred - target) * scale


@torch.no_grad()
def run(dtype: str, shape, scale: float | None):
    device = "cuda"
    m = aicf.Model(dtype=dtype, device=device)

    pred = m.input("pred", aicf.TensorSpec(shape, dtype, device))
    targ = m.input("targ", aicf.TensorSpec(shape, dtype, device))

    g = m.add(aicf.MseGrad("mse", scale=scale), pred, targ)
    m.output("g", g)

    torch_dtype = torch.float16 if dtype == "f16" else torch.float32
    P = torch.randn(*shape, device=device, dtype=torch_dtype).contiguous()
    T = torch.randn(*shape, device=device, dtype=torch_dtype).contiguous()

    numel = P.numel()
    s = (2.0 / float(numel)) if scale is None else float(scale)

    ref = ref_mse_grad(P.float(), T.float(), s)
    if torch_dtype == torch.float16:
        ref = ref.half()

    exe = aicf.CudaExecutor()
    out = exe.run(m, {"pred": P, "targ": T})["g"]

    d = maxabs_delta(out, ref)
    tag = "default" if scale is None else f"scale={s:g}"
    print(f"[{dtype}] shape={tuple(shape)} {tag} max|delta|={d:.3e}")
    return d


def neg_shape():
    try:
        device = "cuda"
        m = aicf.Model(dtype="f32", device=device)
        pred = m.input("pred", aicf.TensorSpec((8, 7), "f32", device))
        targ = m.input("targ", aicf.TensorSpec((8, 8), "f32", device))  # mismatch
        g = m.add(aicf.MseGrad("mse"), pred, targ)
        m.output("g", g)
        print("[NEG shape] unexpected OK")
    except Exception as e:
        print("[NEG shape] ok:", str(e).splitlines()[0])


def neg_dtype():
    try:
        device = "cuda"
        m = aicf.Model(dtype="f32", device=device)
        pred = m.input("pred", aicf.TensorSpec((8, 7), "f32", device))
        targ = m.input("targ", aicf.TensorSpec((8, 7), "f16", device))  # dtype mismatch
        g = m.add(aicf.MseGrad("mse"), pred, targ)
        m.output("g", g)
        print("[NEG dtype] unexpected OK")
    except Exception as e:
        print("[NEG dtype] ok:", str(e).splitlines()[0])


def main():
    torch.manual_seed(0)

    worst = 0.0
    worst = max(worst, run("f32", (64, 256), None))
    worst = max(worst, run("f32", (8, 32, 128), None))
    print("[f32] worst =", worst)

    worst16 = 0.0
    worst16 = max(worst16, run("f16", (64, 256), None))
    worst16 = max(worst16, run("f16", (7, 33, 127), None))
    print("[f16] worst =", worst16)

    # explicit scale
    run("f32", (64, 256), 0.125)
    run("f16", (64, 256), 0.125)

    neg_shape()
    neg_dtype()


if __name__ == "__main__":
    main()
