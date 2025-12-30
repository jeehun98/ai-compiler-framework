from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch

# ------------------------------------------------------------
# Path bootstrap (match your minitest style)
# ------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
REPLAY_N = int(os.environ.get("AICF_REPLAY_N", "20"))
SEED = int(os.environ.get("AICF_SEED", "0"))
WARMUP_RUNS = int(os.environ.get("AICF_WARMUP_RUNS", "2"))

# Optional torch reference (informational only)
TORCH_DEVICE = os.environ.get("TORCH_DEVICE", "cuda")

# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ok(msg: str):
    print(f"[OK] {msg}")

def warn(msg: str):
    print(f"[WARN] {msg}")

def fail(msg: str, code: int = 1):
    print(f"[FAIL] {msg}")
    raise SystemExit(code)

@torch.no_grad()
def snapshot_params(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}

@torch.no_grad()
def max_param_diff(model, snaps) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - snaps[n]).abs().max().item()
        if d > m:
            m = float(d)
    return m

@torch.no_grad()
def loss_like(model, x, t) -> float:
    y = model(x)
    diff = (y.data - t.data)
    return float((diff * diff).mean().detach().cpu().item())

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=== PR1 verify: determinism + torch reference ===")
    print(f"replay_n = {REPLAY_N}, seed = {SEED}, warmup_runs = {WARMUP_RUNS}")

    if not torch.cuda.is_available():
        fail("CUDA not available (torch.cuda.is_available() == False)")

    seed_all(SEED)

    # --------------------------------------------------------
    # Torch reference (forward loss only, informational)
    # --------------------------------------------------------
    torch_loss = None
    try:
        torch.set_grad_enabled(False)

        dev = TORCH_DEVICE
        if dev == "cuda" and not torch.cuda.is_available():
            dev = "cpu"

        model_t = torch.nn.Sequential(
            torch.nn.Linear(8, 8, bias=True),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8, bias=True),
        ).to(device=dev, dtype=torch.float32)

        xt = torch.randn(64, 8, device=dev, dtype=torch.float32)
        tt = torch.randn(64, 8, device=dev, dtype=torch.float32)

        pred = model_t(xt)
        torch_loss = torch.mean((pred - tt) ** 2).item()
        ok(f"Torch forward loss = {torch_loss:.9f} (device={dev})")
    except Exception as e:
        warn(f"Torch reference skipped: {e}")

    # --------------------------------------------------------
    # AICF imports (match v3_adam)
    # --------------------------------------------------------
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

    # --------------------------------------------------------
    # Backend setup (REQUIRED)
    # --------------------------------------------------------
    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()
    ok(f"Backend set: {type(bk).__name__}")

    # reset graph state
    backend.capture_reset()
    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Model / Optim / Data (exactly like v3_adam)
    # --------------------------------------------------------
    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )

    optim = Adam(model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=None)

    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)

    for n, p in model.named_parameters():
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    # --------------------------------------------------------
    # Train step to capture (AICF-only ops path)
    # --------------------------------------------------------
    def train_step_aicf_only():
        # reset grads
        optim.zero_grad()

        y = model(x)
        dY = F.mse_grad(y, t)

        # Capture-safe: accumulate must be False
        autograd_backward(y, grad=dY, accumulate=False)

        # Adam update (StepInc + BiasCorr + AdamStep)
        optim.step_()

    # --------------------------------------------------------
    # Warmup OUTSIDE capture: materialize buffers
    # --------------------------------------------------------
    warmup_capture_safe(train_step=train_step_aicf_only, runs=WARMUP_RUNS, sync=True)
    print("[warmup] loss_like =", loss_like(model, x, t))
    print("[warmup] functional buffers =", functional_buffer_stats())

    # reset graph state after warmup
    backend.capture_reset()
    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Capture
    # --------------------------------------------------------
    backend.capture_begin()
    train_step_aicf_only()
    backend.capture_end()
    torch.cuda.synchronize()
    ok("[capture] done")

    # --------------------------------------------------------
    # Replay determinism: loss AND param step diff must be stable
    # - Primary check: loss_like should be identical run-to-run if fully deterministic
    # - Secondary: param_step_maxdiff should be stable too
    # --------------------------------------------------------
    snaps = snapshot_params(model)

    first_loss = None
    first_diffp = None

    for i in range(REPLAY_N):
        backend.replay()
        torch.cuda.synchronize()

        l = loss_like(model, x, t)
        diffp = max_param_diff(model, snaps)

        # update snapshot for next diff check
        snaps = snapshot_params(model)

        # determinism checks
        if first_loss is None:
            first_loss = l
            first_diffp = diffp
        else:
            if l != first_loss:
                fail(f"Replay determinism broken (loss) at iter {i:02d}: {l:.10f} != {first_loss:.10f}")
            if diffp != first_diffp:
                fail(f"Replay determinism broken (param_step_maxdiff) at iter {i:02d}: {diffp:.6e} != {first_diffp:.6e}")

        print(f"[replay {i:02d}] loss_like={l:.10f} param_step_maxdiff={diffp:.6e}")

    ok(f"Determinism OK: {REPLAY_N} replays bitwise-stable (loss_like={first_loss:.10f}, stepdiff={first_diffp:.6e})")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
