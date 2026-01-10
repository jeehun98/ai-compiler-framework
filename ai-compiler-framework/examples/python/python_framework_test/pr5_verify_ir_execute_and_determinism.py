from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch

# ------------------------------------------------------------
# Path bootstrap
# ------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

# ------------------------------------------------------------
# Config
# ------------------------------------------------------------
REPLAY_N = int(os.environ.get("AICF_REPLAY_N", "20"))
SEED = int(os.environ.get("AICF_SEED", "0"))
WARMUP_RUNS = int(os.environ.get("AICF_WARMUP_RUNS", "2"))
TORCH_DEVICE = os.environ.get("TORCH_DEVICE", "cuda")

IR_DUMP = int(os.environ.get("AICF_IR_DUMP", "1"))
LOWER_DUMP = int(os.environ.get("AICF_LOWER_DUMP", "1"))
TRACE_DUMP = int(os.environ.get("AICF_TRACE_DUMP", "1"))

PRINT_LOSS_EVERY = int(os.environ.get("AICF_PRINT_LOSS_EVERY", "0"))  # 0=never
CHECK_RESTORE = int(os.environ.get("AICF_CHECK_RESTORE", "1"))

RUN_IR_EXEC = int(os.environ.get("AICF_RUN_IR_EXEC", "1"))
IR_EXEC_RUNS = int(os.environ.get("AICF_IR_EXEC_RUNS", "1"))

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
def loss_like(model, x, t) -> float:
    y = model(x)
    diff = (y.data - t.data)
    return float((diff * diff).mean().detach().cpu().item())

@torch.no_grad()
def snapshot_params(model) -> dict:
    snap = {}
    for n, p in list(model.named_parameters()):
        snap[n] = p.data.detach().clone()
    return snap

@torch.no_grad()
def assert_params_changed(before: dict, after: dict, *, tag: str):
    changed = 0
    for n, b in before.items():
        a = after[n]
        if not torch.equal(b, a):
            changed += 1
    if changed == 0:
        raise RuntimeError(f"[{tag}] params did NOT change after step")
    ok(f"[{tag}] params changed: {changed}/{len(before)}")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=== PR5 verify: compile+lower+capture+trace + IRExecutor(IR-only) + determinism(full restore) ===")
    print(f"replay_n={REPLAY_N}, seed={SEED}, warmup_runs={WARMUP_RUNS}, "
          f"ir_exec={RUN_IR_EXEC}, ir_exec_runs={IR_EXEC_RUNS}, print_loss_every={PRINT_LOSS_EVERY}")

    if not torch.cuda.is_available():
        fail("CUDA not available")

    seed_all(SEED)

    # Torch reference (informational)
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

    # AICF imports
    from aicf_fw.backend.aicf_backend import AICFBackend
    from aicf_fw.backend import set_backend, get_backend
    from aicf_fw.core.tensor import Tensor
    from aicf_fw.nn.linear import Linear
    from aicf_fw.nn.relu import ReLU
    from aicf_fw.nn.sequential import Sequential
    from aicf_fw.optim.adam import Adam

    from aicf_fw.core.executor import IRExecutor

    # Backend setup
    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()
    ok(f"Backend set: {type(bk).__name__}")

    backend.capture_reset()
    torch.cuda.synchronize()

    # Model / Optim / Data
    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )
    optim = Adam(model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=None)

    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False, name="x")
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False, name="t")

    for n, p in list(model.named_parameters()):
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    os.environ.setdefault("AICF_ENFORCE_ADAMSTEP_RUNTIME", "1")

    # 1) model.compile
    art = model.compile(
        optim=optim,
        x=x,
        t=t,
        loss="mse",
        name="train_step_aicf_only",
        warmup_runs=WARMUP_RUNS,
        warmup_sync=True,
        validate=True,
        trace=True,
        enforce_ops=("adam_step",),
        torch_sync=True,
    )

    # attach for executor binding
    art.model = model
    art.optim = optim
    art.x = x
    art.t = t

    if IR_DUMP:
        print("=== IR DUMP ===")
        print(art.ir.dump_json(indent=2))

    if LOWER_DUMP:
        print("=== LOWERED OPS ===")
        for i, it in enumerate(art.lowered):
            print(f"[lower {i:02d}] op={it['op']} inputs={it.get('inputs', [])} outputs={it.get('outputs', [])} attrs={it.get('attrs', {})}")

    if TRACE_DUMP:
        print("=== TRACE OPS (runtime capture) ===")
        for i, op in enumerate(art.trace_ops):
            print(f"[trace {i:02d}] op={op}")

    # 2) 기존 체크
    art.assert_runtime_matches_lowering(model, trace_filter=True)
    ok("[lowering] match: forward slice OK, optim slice OK")

    art.assert_adam_state_mutates(model, optim, tag="smoke")
    ok("[adam] state mutation OK on replay")

    # 3) IRExecutor (IR-only) 실행
    if RUN_IR_EXEC:
        exe = IRExecutor.from_artifact(art)

        before = snapshot_params(model)
        for _ in range(IR_EXEC_RUNS):
            exe.run()
        torch.cuda.synchronize()
        after = snapshot_params(model)

        assert_params_changed(before, after, tag="ir_exec")
        ok(f"[ir_exec] IR-only run OK (runs={IR_EXEC_RUNS})")

    # 4) determinism
    art.assert_determinism(
        model,
        optim,
        replays=REPLAY_N,
        check_restore=bool(CHECK_RESTORE),
        print_every=PRINT_LOSS_EVERY,
        loss_fn=(lambda: loss_like(model, x, t)) if PRINT_LOSS_EVERY else None,
        tag="",
    )
    ok(f"Determinism OK: {REPLAY_N} replays stepdiff-sequence matches")

    print("OK.")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
