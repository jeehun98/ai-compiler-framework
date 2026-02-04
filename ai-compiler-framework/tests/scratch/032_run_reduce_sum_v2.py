from __future__ import annotations

import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[1]   # python/aicf_v2
SRC = ROOT / "src"
sp = str(SRC)
if sp not in sys.path:
    sys.path.insert(0, sp)

import aicf_v2 as aicf


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


@torch.no_grad()
def run_f32():
    torch.manual_seed(0)
    device = "cuda"

    M, N = 128, 64
    m = aicf.Model(dtype="f32", device=device)

    dY = m.input("dY", aicf.TensorSpec((M, N), "f32", device))
    dB = m.add(aicf.ReduceSum(name="rsum", axis=0), dY)
    m.output("dB", dB)

    exe = aicf.CudaExecutor()

    t_dY = torch.randn(M, N, device=device, dtype=torch.float32).contiguous()
    out = exe.run(m, {"dY": t_dY})

    ref = t_dY.sum(dim=0)
    d = maxabs_delta(out["dB"], ref)
    print("[F32] max|delta| =", d)


@torch.no_grad()
def run_f16_to_f32():
    torch.manual_seed(1)
    device = "cuda"

    M, N = 257, 64
    m = aicf.Model(dtype="f16", device=device)

    dY = m.input("dY", aicf.TensorSpec((M, N), "f16", device))
    dB = m.add(aicf.ReduceSum(name="rsum", axis=0), dY)
    m.output("dB", dB)

    exe = aicf.CudaExecutor()

    t_dY = torch.randn(M, N, device=device, dtype=torch.float16).contiguous()
    out = exe.run(m, {"dY": t_dY})

    ref = t_dY.float().sum(dim=0)
    d = maxabs_delta(out["dB"], ref)
    print("[F16->F32] max|delta| =", d)


def run_negative_axis1():
    try:
        device = "cuda"
        M, N = 32, 16
        m = aicf.Model(dtype="f32", device=device)
        dY = m.input("dY", aicf.TensorSpec((M, N), "f32", device))

        # should fail at emit time
        dB = m.add(aicf.ReduceSum(name="rsum_bad", axis=1), dY)
        m.output("dB", dB)

        exe = aicf.CudaExecutor()
        t_dY = torch.randn(M, N, device=device, dtype=torch.float32).contiguous()
        exe.run(m, {"dY": t_dY})

        print("[NEG axis=1] ERROR: expected failure but succeeded")
    except Exception as e:
        print("[NEG axis=1] ok:", str(e).splitlines()[0])


def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    print("ReduceSum v2 test")
    run_f32()
    run_f16_to_f32()
    run_negative_axis1()


if __name__ == "__main__":
    main()
