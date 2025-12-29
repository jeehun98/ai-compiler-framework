# examples/python/python_framework_test/minitest_train_replay_v3_adam.py

from __future__ import annotations

import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.backend.aicf_backend import AICFBackend
from aicf_fw.backend import set_backend, get_backend
from aicf_fw.core.tensor import Tensor
from aicf_fw.core.autograd import backward as autograd_backward
from aicf_fw.core.warmup import warmup_capture_safe
from aicf_fw.core.functional import functional_buffer_stats
from aicf_fw.nn.linear import Linear
from aicf_fw.nn.relu import ReLU
from aicf_fw.nn.sequential import Sequential

from aicf_fw.optim.adam import Adam
from aicf_fw.core import functional as F

@torch.no_grad()
def snapshot_params(model: Sequential):
    return {n: p.data.clone() for n, p in model.named_parameters()}

def max_param_diff(model: Sequential, snaps) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - snaps[n]).abs().max().item()
        if d > m:
            m = float(d)
    return m

@torch.no_grad()
def loss_like(model: Sequential, x: Tensor, t: Tensor) -> float:
    y = model(x)
    diff = (y.data - t.data)
    return float((diff * diff).mean().detach().cpu().item())

def main():
    assert torch.cuda.is_available()

    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()

    backend.capture_reset()
    torch.cuda.synchronize()

    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )

    # Adam capture-safe
    optim = Adam(model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=None)

    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)

    for n, p in model.named_parameters():
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    def train_step_aicf_only():
        # Reset grads (now we can safely do it even if we later allow accumulate=True outside capture)
        optim.zero_grad()

        y = model(x)
        dY = F.mse_grad(y, t)  # your existing op-based mse_grad

        # Capture-safe: accumulate must be False (your autograd enforces this)
        autograd_backward(y, grad=dY, accumulate=False)

        # Adam update (StepInc + BiasCorr + AdamStep)
        optim.step_()

    # ----------------------------
    # Warmup: materialize ALL buffers OUTSIDE capture
    # ----------------------------
    warmup_capture_safe(train_step=train_step_aicf_only, runs=2, sync=True)
    print("[warmup] loss_like =", loss_like(model, x, t))
    print("[warmup] functional buffers =", functional_buffer_stats())

    # Reset graph state after warmup
    backend.capture_reset()
    torch.cuda.synchronize()

    # ----------------------------
    # Capture
    # ----------------------------
    backend.capture_begin()
    train_step_aicf_only()
    backend.capture_end()

    torch.cuda.synchronize()
    print("[capture] done")

    snaps = snapshot_params(model)
    for i in range(20):
        backend.replay()
        torch.cuda.synchronize()

        l = loss_like(model, x, t)
        diffp = max_param_diff(model, snaps)
        snaps = snapshot_params(model)
        print(f"[replay {i:02d}] loss_like={l:.10f} param_step_maxdiff={diffp:.6e}")

    print("OK")

if __name__ == "__main__":
    main()
