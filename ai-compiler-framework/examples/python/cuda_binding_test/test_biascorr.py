import os
import sys
from pathlib import Path
import math
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYMOD_DIR = REPO_ROOT / "build" / "python"
PKG_DIR   = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))
if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import aicf_cuda as aicf


def check_close(name, got, ref, atol=1e-6, rtol=1e-6):
    max_abs = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={max_abs} atol={atol} rtol={rtol}")
    if not ok:
        raise RuntimeError(f"{name} mismatch")


def op_call(kind, inputs, outputs, attrs=None):
    if attrs is None:
        attrs = {}
    aicf.op_call(kind, inputs, outputs, attrs)


@torch.inference_mode()
def test_biascorr_matches_cpu(beta1=0.9, beta2=0.999, steps=(1, 2, 3, 10, 100, 1000)):
    assert torch.cuda.is_available()

    step = torch.empty((), device="cuda", dtype=torch.int32)
    bc1  = torch.empty((), device="cuda", dtype=torch.float32)
    bc2  = torch.empty((), device="cuda", dtype=torch.float32)

    for t in steps:
        step.fill_(t)
        op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})
        torch.cuda.synchronize()

        # CPU ref
        b1t = beta1 ** t
        b2t = beta2 ** t
        ref1 = 1.0 / (1.0 - b1t)
        ref2 = 1.0 / (1.0 - b2t)

        ref = torch.tensor([ref1, ref2], device="cuda", dtype=torch.float32)
        got = torch.stack([bc1, bc2])

        check_close(f"BiasCorr(t={t})", got, ref, atol=1e-6, rtol=1e-6)


@torch.inference_mode()
def test_biascorr_with_stepinc(beta1=0.9, beta2=0.999, iters=50):
    """
    step=0에서 시작해서 매번 StepInc 후 BiasCorr 호출.
    """
    step = torch.zeros((), device="cuda", dtype=torch.int32)
    bc1  = torch.empty((), device="cuda", dtype=torch.float32)
    bc2  = torch.empty((), device="cuda", dtype=torch.float32)

    for i in range(iters):
        op_call(aicf.OpKind.StepInc, [step], [step], {})
        op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})
    torch.cuda.synchronize()

    t = int(step.cpu().item())
    assert t == iters, f"expected step={iters}, got {t}"

    # last value check
    ref1 = 1.0 / (1.0 - (beta1 ** t))
    ref2 = 1.0 / (1.0 - (beta2 ** t))
    ref = torch.tensor([ref1, ref2], device="cuda", dtype=torch.float32)
    got = torch.stack([bc1, bc2])
    check_close(f"BiasCorr(StepInc loop t={t})", got, ref, atol=1e-6, rtol=1e-6)


@torch.inference_mode()
def test_biascorr_cudagraph_capture_safe(beta1=0.9, beta2=0.999, replays=200):
    """
    torch.cuda.CUDAGraph 캡처/리플레이 안전성.
    주의: CUDAGraph는 non-default stream에서 capture_begin 해야 함.
    """
    step = torch.zeros((), device="cuda", dtype=torch.int32)
    bc1  = torch.empty((), device="cuda", dtype=torch.float32)
    bc2  = torch.empty((), device="cuda", dtype=torch.float32)

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.StepInc, [step], [step], {})
        op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})
    torch.cuda.synchronize()

    s = torch.cuda.Stream()
    g = torch.cuda.CUDAGraph()

    torch.cuda.synchronize()
    with torch.cuda.stream(s):
        g.capture_begin()
        op_call(aicf.OpKind.StepInc, [step], [step], {})
        op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})
        g.capture_end()
    torch.cuda.synchronize()

    for _ in range(replays):
        g.replay()
    torch.cuda.synchronize()

    print("[BiasCorr(CUDAGraph)] replay OK (no crash)")


@torch.inference_mode()
def test_biascorr_aicf_graph_capture_safe(beta1=0.9, beta2=0.999, replays=500):
    """
    AICF 전용 capture_begin/end/replay (dedicated stream) 캡처/리플레이 안전성.
    """
    step = torch.zeros((), device="cuda", dtype=torch.int32)
    bc1  = torch.empty((), device="cuda", dtype=torch.float32)
    bc2  = torch.empty((), device="cuda", dtype=torch.float32)

    # warmup
    for _ in range(10):
        op_call(aicf.OpKind.StepInc, [step], [step], {})
        op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})
    torch.cuda.synchronize()

    aicf.capture_reset()
    torch.cuda.synchronize()

    aicf.capture_begin()
    op_call(aicf.OpKind.StepInc, [step], [step], {})
    op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})
    aicf.capture_end()
    torch.cuda.synchronize()

    for _ in range(replays):
        aicf.replay()
    torch.cuda.synchronize()

    print("[BiasCorr(AICF Graph)] replay OK (no crash)")


if __name__ == "__main__":
    test_biascorr_matches_cpu()
    test_biascorr_with_stepinc()
    test_biascorr_cudagraph_capture_safe()
    test_biascorr_aicf_graph_capture_safe()
    print("ALL OK")
