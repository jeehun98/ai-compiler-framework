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


def check_close(name, got, ref, atol=2e-5, rtol=2e-5):
    m = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={m} atol={atol} rtol={rtol}")
    if not ok:
        raise RuntimeError(f"{name} mismatch")


@torch.inference_mode()
def test_adam_biascorr_tensor_inputs_matches_torch_and_traces(steps=20, shape=(1024,)):
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    p  = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    g0 = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    m  = torch.zeros_like(p)
    v  = torch.zeros_like(p)

    step = torch.zeros((), device="cuda", dtype=torch.int32)
    bc1  = torch.empty((), device="cuda", dtype=torch.float32)
    bc2  = torch.empty((), device="cuda", dtype=torch.float32)

    # torch reference
    p_ref = p.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([p_ref], lr=lr, betas=(beta1, beta2), eps=eps)

    aicf.trace_reset()
    aicf.trace_enable(True)

    for _ in range(steps):
        # torch
        p_ref.grad = g0.detach().clone()
        opt.step()

        # AICF
        op_call(aicf.OpKind.StepInc,  [step], [step], {})
        op_call(aicf.OpKind.BiasCorr, [step], [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})

        op_call(
            aicf.OpKind.AdamStep,
            [p, g0, m, v, bc1, bc2],
            [p, m, v],
            {"lr": float(lr), "beta1": float(beta1), "beta2": float(beta2), "eps": float(eps)},
        )

    torch.cuda.synchronize()

    # 핵심: adam_step이 실제로 op_call을 탔는지
    trace = aicf.trace_get()
    n_adam = sum(1 for x in trace if x == "adam_step")
    print("[trace] total_ops=", len(trace), "adam_step=", n_adam)
    assert n_adam == steps, f"adam_step trace mismatch: expected {steps}, got {n_adam}"

    check_close("AdamStep(v1 bc-tensor) vs torch", p, p_ref.detach(), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    test_adam_biascorr_tensor_inputs_matches_torch_and_traces(steps=20, shape=(1024,))
    print("OK")
