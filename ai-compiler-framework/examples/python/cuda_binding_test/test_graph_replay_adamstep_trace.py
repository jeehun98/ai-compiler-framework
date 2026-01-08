import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYMOD_DIR = REPO_ROOT / "build" / "python"
PKG_DIR   = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))
if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import aicf_cuda as aicf


def op_call(kind, inputs, outputs, attrs=None):
    if attrs is None:
        attrs = {}
    aicf.op_call(kind, inputs, outputs, attrs)


@torch.inference_mode()
def test_adam_step_full_aicf_graph_replay_and_traces(replays=200, shape=(4096,)):
    assert torch.cuda.is_available()
    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    p  = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    g0 = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    m  = torch.zeros_like(p)
    v  = torch.zeros_like(p)

    step = torch.zeros((), device="cuda", dtype=torch.int32)
    bc1  = torch.empty((), device="cuda", dtype=torch.float32)
    bc2  = torch.empty((), device="cuda", dtype=torch.float32)

    attrs_adam = {"lr": float(lr), "beta1": float(beta1), "beta2": float(beta2), "eps": float(eps)}
    attrs_bc   = {"beta1": float(beta1), "beta2": float(beta2)}

    # warmup
    for _ in range(3):
        op_call(aicf.OpKind.StepInc,  [step], [step], {})
        op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], attrs_bc)
        op_call(aicf.OpKind.AdamStep, [p, g0, m, v, bc1, bc2], [p, m, v], attrs_adam)
    torch.cuda.synchronize()

    aicf.capture_reset()
    torch.cuda.synchronize()

    # capture
    aicf.capture_begin()
    op_call(aicf.OpKind.StepInc,  [step], [step], {})
    op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], attrs_bc)
    op_call(aicf.OpKind.AdamStep, [p, g0, m, v, bc1, bc2], [p, m, v], attrs_adam)
    aicf.capture_end()
    torch.cuda.synchronize()

    # trace should include adam_step once (only capture region counted due to trace_reset at capture_begin)
    trace = aicf.trace_get()
    print("[trace@capture] ", trace)
    assert trace.count("adam_step") == 1, "adam_step not captured in trace"
    assert trace == ["step_inc", "bias_corr", "adam_step"], "unexpected op order in capture trace"

    # replay
    for _ in range(replays):
        aicf.replay()
    torch.cuda.synchronize()

    print("[AICF Graph] StepInc+BiasCorr+AdamStep replay OK (no crash)")


if __name__ == "__main__":
    test_adam_step_full_aicf_graph_replay_and_traces(replays=200, shape=(4096,))
    print("OK")
