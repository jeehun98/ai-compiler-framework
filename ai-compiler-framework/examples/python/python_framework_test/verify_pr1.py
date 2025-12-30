# ============================================================
# PR1 verification script (single run) - sequence determinism
#
# What it checks (PR1-appropriate):
#   - Build + warmup + capture once, then replay N steps => collect:
#       * loss_in_step[i] (computed inside captured step)
#       * params snapshot after the N replays
#   - Reset graph state, re-run the exact same procedure again => collect again
#   - PASS if:
#       * loss sequence is bitwise identical across the two runs
#       * final params are bitwise identical across the two runs
#
# This matches "determinism of replay sequence" for stateful optimizers (Adam).
#
# Run:
#   python examples/python/python_framework_test/verify_pr1.py
#
# Env:
#   AICF_REPLAY_N=20
#   AICF_SEED=0
#   AICF_WARMUP_RUNS=2
# ============================================================

from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch

# ------------------------------------------------------------
# Path bootstrap (match minitest style)
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

# Optional torch reference (informational)
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
    # clone to freeze values
    return {n: p.data.clone() for n, p in model.named_parameters()}

@torch.no_grad()
def max_param_absdiff(snaps_a: dict, snaps_b: dict) -> tuple[str, float]:
    worst_name = ""
    worst = 0.0
    for n in snaps_a.keys():
        d = (snaps_a[n] - snaps_b[n]).abs().max().item()
        if d > worst:
            worst = float(d)
            worst_name = n
    return worst_name, worst

def run_once(tag: str):
    """
    One full PR1 run:
      - set backend
      - reset capture state
      - build model/optim/data (deterministic under seed_all)
      - warmup outside capture
      - capture 1 step
      - replay N steps, recording loss_in_step sequence
      - return (loss_seq:list[float], final_param_snaps:dict[name->tensor])
    """
    print(f"\n=== RUN {tag} ===")

    # AICF imports (aligned with v3_adam)
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

    # Backend setup
    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()
    ok(f"[{tag}] Backend set: {type(bk).__name__}")

    # reset graph state
    backend.capture_reset()
    torch.cuda.synchronize()

    # Build model/optim/data
    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )
    optim = Adam(model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=None)

    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False)

    for n, p in model.named_parameters():
        print(f"[{tag}][param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    # loss scalar inside captured step
    last_loss = {"v": None}

    def train_step_aicf_only():
        optim.zero_grad()

        y = model(x)

        # loss computed inside step (torch ops)
        diff = (y.data - t.data)
        last_loss["v"] = (diff * diff).mean()

        dY = F.mse_grad(y, t)
        autograd_backward(y, grad=dY, accumulate=False)

        optim.step_()

    # Warmup outside capture
    warmup_capture_safe(train_step=train_step_aicf_only, runs=WARMUP_RUNS, sync=True)
    torch.cuda.synchronize()
    if last_loss["v"] is None:
        fail(f"[{tag}] last_loss not produced during warmup")
    print(f"[{tag}][warmup] loss_in_step =", float(last_loss["v"].detach().cpu().item()))
    print(f"[{tag}][warmup] functional buffers =", functional_buffer_stats())

    # Reset after warmup (same as v3_adam)
    backend.capture_reset()
    torch.cuda.synchronize()

    # Capture 1 step
    backend.capture_begin()
    train_step_aicf_only()
    backend.capture_end()
    torch.cuda.synchronize()
    ok(f"[{tag}] capture done")

    # Replay sequence
    loss_seq = []
    for i in range(REPLAY_N):
        backend.replay()
        torch.cuda.synchronize()

        if last_loss["v"] is None:
            fail(f"[{tag}] last_loss is None during replay {i}")

        l = float(last_loss["v"].detach().cpu().item())
        loss_seq.append(l)
        print(f"[{tag}][replay {i:02d}] loss_in_step={l:.10f}")

    final_snaps = snapshot_params(model)
    ok(f"[{tag}] captured loss_seq len={len(loss_seq)} and final params snapshot")
    return loss_seq, final_snaps

def main():
    print("=== PR1 verify: sequence determinism (two full runs compare) ===")
    print(f"replay_n = {REPLAY_N}, seed = {SEED}, warmup_runs = {WARMUP_RUNS}")

    if not torch.cuda.is_available():
        fail("CUDA not available (torch.cuda.is_available() == False)")

    # Torch informational reference
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
    # Two identical runs under the same seed
    # --------------------------------------------------------
    seed_all(SEED)
    loss_a, snaps_a = run_once("A")

    seed_all(SEED)
    loss_b, snaps_b = run_once("B")

    # Compare loss sequences bitwise (exact float compare)
    if len(loss_a) != len(loss_b):
        fail(f"loss_seq length mismatch: {len(loss_a)} vs {len(loss_b)}")

    for i, (la, lb) in enumerate(zip(loss_a, loss_b)):
        if la != lb:
            fail(f"Sequence determinism broken (loss_seq) at idx {i}: {la:.10f} != {lb:.10f}")

    ok("loss_seq bitwise identical across runs")

    # Compare final params bitwise
    worst_name, worst = max_param_absdiff(snaps_a, snaps_b)
    if worst != 0.0:
        fail(f"Sequence determinism broken (final params): worst={worst_name} max_absdiff={worst:.6e}")

    ok("final params bitwise identical across runs")
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
