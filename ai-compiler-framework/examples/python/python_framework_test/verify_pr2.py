# ============================================================
# PR2 verification script - training progress + sequence determinism
#
# Checks:
#  1) Training progresses:
#     - params change across replays
#     - Adam state (step, m/v) changes
#     - loss typically changes (not required to strictly monotonic)
#  2) Sequence determinism:
#     - Run A and Run B under same seed produce identical sequences:
#         * loss_in_step[i] bitwise
#         * param snapshots at selected checkpoints bitwise
#         * adam step tensor value identical
#
# Run:
#   python examples/python/python_framework_test/verify_pr2.py
#
# Env:
#   AICF_REPLAY_N=50
#   AICF_SEED=0
#   AICF_WARMUP_RUNS=2
#   AICF_CHECKPOINTS=0,1,2,4,8,16,32,49   (comma-separated)
# ============================================================

from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

REPLAY_N = int(os.environ.get("AICF_REPLAY_N", "50"))
SEED = int(os.environ.get("AICF_SEED", "0"))
WARMUP_RUNS = int(os.environ.get("AICF_WARMUP_RUNS", "2"))
CKPTS_RAW = os.environ.get("AICF_CHECKPOINTS", "")

def _parse_ckpts(n: int, s: str):
    if not s.strip():
        # default checkpoints
        base = [0, 1, 2, 4, 8, 16, 32, n - 1]
        out = []
        for x in base:
            if 0 <= x < n and x not in out:
                out.append(x)
        return out
    out = []
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        i = int(tok)
        if i < 0:
            i = n + i
        if 0 <= i < n and i not in out:
            out.append(i)
    return sorted(out)

CHECKPOINTS = _parse_ckpts(REPLAY_N, CKPTS_RAW)

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ok(msg: str): print(f"[OK] {msg}")
def warn(msg: str): print(f"[WARN] {msg}")
def fail(msg: str, code: int = 1):
    print(f"[FAIL] {msg}")
    raise SystemExit(code)

@torch.no_grad()
def snapshot_params(model):
    return {n: p.data.clone() for n, p in model.named_parameters()}

@torch.no_grad()
def max_param_absdiff(a: dict, b: dict) -> tuple[str, float]:
    worst_name, worst = "", 0.0
    for n in a.keys():
        d = (a[n] - b[n]).abs().max().item()
        if d > worst:
            worst = float(d)
            worst_name = n
    return worst_name, worst

@torch.no_grad()
def max_param_delta(model, prev_snap: dict) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - prev_snap[n]).abs().max().item()
        if d > m:
            m = float(d)
    return m

def run_once(tag: str):
    """
    Full training sequence run:
      - warmup, capture one step, replay N steps
      - record:
          loss_seq[i] (loss computed inside captured step)
          step_seq[i] (adam step value after step_inc)
          max_param_delta_seq[i] (max abs param change from previous iter)  [for progress logging]
          param_snaps at checkpoints (bitwise)
    """
    print(f"\n=== RUN {tag} ===")

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

    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()
    ok(f"[{tag}] Backend set: {type(bk).__name__}")

    backend.capture_reset()
    torch.cuda.synchronize()

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

    last_loss = {"v": None}

    # Try to read Adam step if available.
    # We won't fail if we can't find it (depends on your Adam implementation).
    def _read_step():
        # common patterns:
        #  - optim.step (python int)   OR
        #  - optim.step_tensor.data (torch scalar) OR
        #  - optim.state["step"].data ...
        for attr in ["step_tensor", "step"]:
            if hasattr(optim, attr):
                v = getattr(optim, attr)
                if isinstance(v, int):
                    return int(v)
                if hasattr(v, "data"):  # Tensor
                    td = v.data
                    if torch.is_tensor(td) and td.numel() == 1:
                        return int(td.detach().cpu().item())
                if torch.is_tensor(v) and v.numel() == 1:
                    return int(v.detach().cpu().item())
        # fallback: scan dict-like state
        if hasattr(optim, "state") and isinstance(optim.state, dict):
            for k, v in optim.state.items():
                if k in ("step", "t") and hasattr(v, "data"):
                    td = v.data
                    if torch.is_tensor(td) and td.numel() == 1:
                        return int(td.detach().cpu().item())
        return None

    def train_step_aicf_only():
        optim.zero_grad()
        y = model(x)

        diff = (y.data - t.data)
        last_loss["v"] = (diff * diff).mean()

        dY = F.mse_grad(y, t)
        autograd_backward(y, grad=dY, accumulate=False)

        optim.step_()

    # warmup
    warmup_capture_safe(train_step=train_step_aicf_only, runs=WARMUP_RUNS, sync=True)
    torch.cuda.synchronize()
    if last_loss["v"] is None:
        fail(f"[{tag}] last_loss not produced during warmup")
    print(f"[{tag}][warmup] loss_in_step =", float(last_loss["v"].detach().cpu().item()))
    print(f"[{tag}][warmup] functional buffers =", functional_buffer_stats())

    backend.capture_reset()
    torch.cuda.synchronize()

    # capture
    backend.capture_begin()
    train_step_aicf_only()
    backend.capture_end()
    torch.cuda.synchronize()
    ok(f"[{tag}] capture done")

    # replay N times
    loss_seq = []
    step_seq = []
    maxdelta_seq = []
    ckpt_snaps = {}  # iter -> param snapshot dict

    prev = snapshot_params(model)

    for i in range(REPLAY_N):
        backend.replay()
        torch.cuda.synchronize()

        if last_loss["v"] is None:
            fail(f"[{tag}] last_loss is None during replay {i}")

        l = float(last_loss["v"].detach().cpu().item())
        loss_seq.append(l)

        md = max_param_delta(model, prev)
        maxdelta_seq.append(md)

        st = _read_step()
        step_seq.append(st)

        # checkpoint snapshots (bitwise compare)
        if i in CHECKPOINTS:
            ckpt_snaps[i] = snapshot_params(model)

        # update prev
        prev = snapshot_params(model)

        print(f"[{tag}][replay {i:02d}] loss={l:.10f} maxΔparam={md:.6e} step={st}")

    # progress assertions (within the run)
    # - params should change at least sometimes
    changed = any(d > 0.0 for d in maxdelta_seq)
    if not changed:
        fail(f"[{tag}] Training did not change params at all (all maxΔparam==0)")

    # - Adam step should advance if we could read it
    if step_seq[0] is not None:
        if step_seq[-1] == step_seq[0]:
            fail(f"[{tag}] Adam step did not change (step stayed {step_seq[0]})")

    ok(f"[{tag}] progress OK (params changed; step advanced if observable)")
    return {
        "loss_seq": loss_seq,
        "step_seq": step_seq,
        "maxdelta_seq": maxdelta_seq,
        "ckpt_snaps": ckpt_snaps,
    }

def main():
    print("=== PR2 verify: training progress + sequence determinism ===")
    print(f"replay_n = {REPLAY_N}, seed = {SEED}, warmup_runs = {WARMUP_RUNS}")
    print(f"checkpoints = {CHECKPOINTS}")

    if not torch.cuda.is_available():
        fail("CUDA not available")

    # Run A
    seed_all(SEED)
    out_a = run_once("A")

    # Run B
    seed_all(SEED)
    out_b = run_once("B")

    # Determinism checks across runs
    la, lb = out_a["loss_seq"], out_b["loss_seq"]
    if len(la) != len(lb):
        fail(f"loss_seq length mismatch: {len(la)} vs {len(lb)}")

    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            fail(f"Sequence determinism broken (loss_seq) at idx {i}: {x:.10f} != {y:.10f}")

    ok("loss_seq bitwise identical across runs")

    # Compare step sequence if observable (ignore None entries)
    sa, sb = out_a["step_seq"], out_b["step_seq"]
    if len(sa) == len(sb) and all(v is not None for v in sa) and all(v is not None for v in sb):
        for i, (x, y) in enumerate(zip(sa, sb)):
            if x != y:
                fail(f"Sequence determinism broken (step_seq) at idx {i}: {x} != {y}")
        ok("step_seq identical across runs (observable)")

    # Compare checkpoint parameter snapshots bitwise
    ca, cb = out_a["ckpt_snaps"], out_b["ckpt_snaps"]
    for i in CHECKPOINTS:
        if i not in ca or i not in cb:
            fail(f"Missing checkpoint snapshot at iter {i}")

        worst_name, worst = max_param_absdiff(ca[i], cb[i])
        if worst != 0.0:
            fail(f"Sequence determinism broken (params@{i}): worst={worst_name} max_absdiff={worst:.6e}")

    ok("checkpoint params bitwise identical across runs")
    print("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
