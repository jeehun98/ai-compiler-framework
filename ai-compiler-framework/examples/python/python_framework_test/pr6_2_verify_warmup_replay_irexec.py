from __future__ import annotations

import os, sys
from pathlib import Path
import random
import numpy as np
import torch


# ------------------------------------------------------------
# Path bootstrap (same style as your previous tests)
# ------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, "../../../.."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


# ------------------------------------------------------------
# Config (env)
# ------------------------------------------------------------
REPLAY_N = int(os.environ.get("AICF_REPLAY_N", "10"))
SEED = int(os.environ.get("AICF_SEED", "0"))
WARMUP_RUNS = int(os.environ.get("AICF_WARMUP_RUNS", "2"))
IR_EXEC_RUNS = int(os.environ.get("AICF_IR_EXEC_RUNS", "1"))
PRINT_EVERY = int(os.environ.get("AICF_PRINT_EVERY", "0"))  # 0=never

# tolerance for replay vs irexec delta check
ATOL = float(os.environ.get("AICF_ATOL", "0.0"))
RTOL = float(os.environ.get("AICF_RTOL", "0.0"))


def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ok(msg: str):
    print(f"[OK] {msg}")


def fail(msg: str, code: int = 1):
    print(f"[FAIL] {msg}")
    raise SystemExit(code)


@torch.no_grad()
def snapshot_named_params(model) -> dict[str, torch.Tensor]:
    return {n: p.data.detach().clone() for n, p in model.named_parameters()}


@torch.no_grad()
def max_param_diff(model, snap: dict[str, torch.Tensor]) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - snap[n]).abs().max().item()
        m = max(m, float(d))
    return m


@torch.no_grad()
def assert_params_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], *, tag: str):
    changed = 0
    for n, b in before.items():
        a = after[n]
        if not torch.equal(b, a):
            changed += 1
    if changed == 0:
        raise RuntimeError(f"[{tag}] params did NOT change")
    ok(f"[{tag}] params changed: {changed}/{len(before)}")


@torch.no_grad()
def assert_params_unchanged(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], *, tag: str):
    changed = 0
    for n, b in before.items():
        a = after[n]
        if not torch.equal(b, a):
            changed += 1
    if changed != 0:
        raise RuntimeError(f"[{tag}] params CHANGED unexpectedly: {changed}/{len(before)}")
    ok(f"[{tag}] params unchanged: {len(before) - changed}/{len(before)}")


@torch.no_grad()
def tensor_maxabs(t: torch.Tensor) -> float:
    return float(t.detach().abs().max().item()) if t.numel() else 0.0


def main():
    print("=== PR6.2 verify: warmup(no-drift) + CUDA-graph replay vs IRExecutor eager ===")
    print(f"seed={SEED}, warmup_runs={WARMUP_RUNS}, replay_n={REPLAY_N}, ir_exec_runs={IR_EXEC_RUNS}")
    print(f"atol={ATOL} rtol={RTOL}")

    if not torch.cuda.is_available():
        fail("CUDA not available")

    seed_all(SEED)
    torch.set_grad_enabled(False)

    # AICF imports
    from aicf_fw.backend.aicf_backend import AICFBackend
    from aicf_fw.backend import set_backend, get_backend
    from aicf_fw.core.autograd import Tensor
    from aicf_fw.nn.linear import Linear
    from aicf_fw.nn.relu import ReLU
    from aicf_fw.nn.sequential import Sequential
    from aicf_fw.optim.adam import Adam
    from aicf_fw.core.runtime import IRExecutor
    from aicf_fw.core.train_state import TrainState

    # backend
    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()
    ok(f"Backend set: {type(bk).__name__}")

    backend.capture_reset()
    torch.cuda.synchronize()

    # model/optim/data
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

    # enforce adam_step dispatch in runtime (if you use this flag downstream)
    os.environ.setdefault("AICF_ENFORCE_ADAMSTEP_RUNTIME", "1")

    # ------------------------------------------------------------
    # (0) Snapshot before compile: to verify warmup drift-free
    # ------------------------------------------------------------
    st_pre = TrainState.capture(model, optim)
    snap_pre = snapshot_named_params(model)

    # ------------------------------------------------------------
    # compile+capture
    # ------------------------------------------------------------
    art = model.compile(
        optim=optim,
        x=x,
        t=t,
        loss="mse",
        name="train_step_pr6_2",
        warmup_runs=WARMUP_RUNS,
        warmup_sync=True,
        validate=True,
        trace=True,
        enforce_ops=("adam_step",),
        torch_sync=True,
    )

    # (1) Warmup drift-free check
    # compile path executed warmup internally. 상태가 그대로여야 함.
    st_post = TrainState.capture(model, optim)
    snap_post = snapshot_named_params(model)

    if int(st_post.step.item()) != int(st_pre.step.item()):
        fail(f"[warmup] step drift: {int(st_pre.step.item())} -> {int(st_post.step.item())}")
    assert_params_unchanged(snap_pre, snap_post, tag="warmup_no_drift")
    ok(f"[warmup_no_drift] step stays {int(st_post.step.item())}")

    # (2) trace vs lowering coverage
    art.assert_runtime_matches_lowering(model, trace_filter=True)
    ok("[lowering/trace] match OK")

    # ------------------------------------------------------------
    # (A) Replay mutates state
    # ------------------------------------------------------------
    st0 = TrainState.capture(model, optim)
    snap0 = snapshot_named_params(model)

    art.backend.replay()
    torch.cuda.synchronize()

    st1 = TrainState.capture(model, optim)
    snap1 = snapshot_named_params(model)

    if int(st1.step.item()) == int(st0.step.item()):
        fail("replay did not advance step")
    assert_params_changed(snap0, snap1, tag="replay_once")
    ok(f"[replay_once] step {int(st0.step.item())} -> {int(st1.step.item())}")

    # ------------------------------------------------------------
    # (B) IRExecutor eager run mutates params too (restore first)
    # ------------------------------------------------------------
    st0.restore(model, optim)
    torch.cuda.synchronize()

    exe = IRExecutor.from_artifact(art)

    snap2 = snapshot_named_params(model)
    for _ in range(IR_EXEC_RUNS):
        exe.run()
    torch.cuda.synchronize()
    snap3 = snapshot_named_params(model)
    assert_params_changed(snap2, snap3, tag="ir_exec")
    ok(f"[ir_exec] runs={IR_EXEC_RUNS}")

    # ------------------------------------------------------------
    # (C) Replay vs IRExec single-step delta compare (tolerance)
    # ------------------------------------------------------------
    # 기준 상태를 하나 잡고,
    #  - replay 1회 후 delta
    #  - restore 후 irexec 1회 후 delta
    # 두 delta가 유사해야 함.
    st_base = TrainState.capture(model, optim)

    # replay delta
    st_base.restore(model, optim)
    torch.cuda.synchronize()
    snap_r0 = snapshot_named_params(model)
    art.backend.replay()
    torch.cuda.synchronize()
    delta_replay = max_param_diff(model, snap_r0)

    # irexec delta
    st_base.restore(model, optim)
    torch.cuda.synchronize()
    snap_i0 = snapshot_named_params(model)
    exe.run()
    torch.cuda.synchronize()
    delta_irexec = max_param_diff(model, snap_i0)

    # compare
    diff = abs(delta_replay - delta_irexec)
    tol = ATOL + RTOL * max(abs(delta_replay), abs(delta_irexec))
    if diff > tol:
        fail(
            "[replay_vs_irexec] delta mismatch "
            f"replay={delta_replay:.6e} irexec={delta_irexec:.6e} diff={diff:.6e} tol={tol:.6e}"
        )
    ok(
        f"[replay_vs_irexec] delta close: replay={delta_replay:.6e} irexec={delta_irexec:.6e} "
        f"(diff={diff:.3e} tol={tol:.3e})"
    )

    # ------------------------------------------------------------
    # (D) Determinism: replay stepdiff sequence matches after restore
    # ------------------------------------------------------------
    st_base.restore(model, optim)
    torch.cuda.synchronize()

    A: list[float] = []
    snaps = snapshot_named_params(model)
    for i in range(REPLAY_N):
        art.backend.replay()
        torch.cuda.synchronize()
        A.append(max_param_diff(model, snaps))
        snaps = snapshot_named_params(model)
        if PRINT_EVERY and (i % PRINT_EVERY == 0):
            print(f"[A] i={i:02d} stepdiff={A[-1]:.6e}")

    st_base.restore(model, optim)
    torch.cuda.synchronize()

    B: list[float] = []
    snaps = snapshot_named_params(model)
    for i in range(REPLAY_N):
        art.backend.replay()
        torch.cuda.synchronize()
        B.append(max_param_diff(model, snaps))
        snaps = snapshot_named_params(model)
        if PRINT_EVERY and (i % PRINT_EVERY == 0):
            print(f"[B] i={i:02d} stepdiff={B[-1]:.6e}")

    for i, (a, b) in enumerate(zip(A, B)):
        if a != b:
            fail(f"determinism broken at iter {i:02d}: {a:.6e} != {b:.6e}")

    ok(f"Determinism OK: {REPLAY_N} replays stepdiff-sequence matches")
    print("OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
