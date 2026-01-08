import os
import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
REPO_ROOT = THIS.parents[3]
EXAMPLES_PY = THIS.parents[1]

PYMOD_DIR = REPO_ROOT / "build" / "python"
PKG_DIR   = PYMOD_DIR / "aicf_cuda"

if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

sys.path.insert(0, str(PYMOD_DIR))
if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import aicf_cuda as aicf

from aicf_fw.backend.aicf_backend import AICFBackend
from aicf_fw.backend import set_backend
from aicf_fw.core.tensor import Tensor
from aicf_fw.core import functional as F


def check_close(name, got, ref, atol=2e-5, rtol=2e-5):
    m = (got - ref).abs().max().item()
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={m} atol={atol} rtol={rtol}")
    if not ok:
        raise RuntimeError(f"{name} mismatch")


@torch.inference_mode()
def test_functional_adam_step_runtime_dispatch_and_matches_torch(steps=20, shape=(1024,)):
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    bk = AICFBackend()
    set_backend(bk)

    lr, beta1, beta2, eps = 1e-3, 0.9, 0.999, 1e-8

    p  = Tensor(torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous(), requires_grad=False, name="p")
    g0 = Tensor(torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous(), requires_grad=False, name="g")
    m  = Tensor(torch.zeros_like(p.data), requires_grad=False, name="m")
    v  = Tensor(torch.zeros_like(p.data), requires_grad=False, name="v")

    step = torch.zeros((), device="cuda", dtype=torch.int32)
    bc1  = torch.empty((), device="cuda", dtype=torch.float32)
    bc2  = torch.empty((), device="cuda", dtype=torch.float32)

    # torch reference
    p_ref = p.data.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([p_ref], lr=lr, betas=(beta1, beta2), eps=eps)

    aicf.trace_reset()
    aicf.trace_enable(True)

    for _ in range(steps):
        p_ref.grad = g0.data.detach().clone()
        opt.step()

        F.step_inc_(step)
        F.bias_corr_out(step, bc1, bc2, beta1, beta2)
        F.adam_step_(p=p, g=g0, m=m, v=v, bc1_inv=bc1, bc2_inv=bc2,
                     lr=lr, beta1=beta1, beta2=beta2, eps=eps)

    torch.cuda.synchronize()

    trace = aicf.trace_get()
    n_adam = trace.count("adam_step")
    print("[trace] total_ops=", len(trace), "adam_step=", n_adam)
    assert n_adam == steps, f"adam_step trace mismatch: expected {steps}, got {n_adam}"

    check_close("functional AdamStep vs torch", p.data, p_ref.detach(), atol=2e-5, rtol=2e-5)


if __name__ == "__main__":
    os.environ.setdefault("AICF_ENFORCE_ADAMSTEP_RUNTIME", "0")  # 여기선 굳이 강제 안 해도 됨
    test_functional_adam_step_runtime_dispatch_and_matches_torch(steps=20, shape=(1024,))
    print("OK")
