from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch


# ------------------------------------------------------------
# Path bootstrap (aicf_fw lives under examples/python)
# ------------------------------------------------------------
THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))


# ------------------------------------------------------------
# Config (env)
# ------------------------------------------------------------
SEED = int(os.environ.get("AICF_SEED", "0"))
WARMUP_RUNS = int(os.environ.get("AICF_WARMUP_RUNS", "2"))
REPLAY_N = int(os.environ.get("AICF_REPLAY_N", "10"))
IR_EXEC_RUNS = int(os.environ.get("AICF_IR_EXEC_RUNS", "1"))
PRINT_EVERY = int(os.environ.get("AICF_PRINT_EVERY", "0"))  # 0=never

# allclose 기준 기본값 (strict 금지)
EQ_ATOL = float(os.environ.get("AICF_EQ_ATOL", "1e-4"))
EQ_RTOL = float(os.environ.get("AICF_EQ_RTOL", "1e-4"))

# Debug: print non-finite stats if happens
DEBUG_NAN = int(os.environ.get("AICF_DEBUG_NAN", "1"))  # 1=on, 0=off


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
def snapshot_named_params(model) -> dict[str, torch.Tensor]:
    return {n: p.data.detach().clone() for n, p in model.named_parameters()}


@torch.no_grad()
def _tensor_nan_inf_stats(t: torch.Tensor) -> tuple[int, int, int]:
    nan = int(torch.isnan(t).sum().item())
    inf = int(torch.isinf(t).sum().item())
    return int(t.numel()), nan, inf


@torch.no_grad()
def assert_params_finite(snap: dict[str, torch.Tensor], *, tag: str):
    bad = []
    for k, v in snap.items():
        _, n_nan, n_inf = _tensor_nan_inf_stats(v)
        if n_nan or n_inf:
            bad.append((k, n_nan, n_inf))

    if bad:
        lines = "\n".join([f"  - {k}: nan={nn}, inf={ni}" for (k, nn, ni) in bad[:32]])
        fail(f"[{tag}] non-finite params detected:\n{lines}")


@torch.no_grad()
def max_abs_param_diff_nan_safe(a: dict[str, torch.Tensor], b: dict[str, torch.Tensor]) -> float:
    """
    NaN-safe max abs diff:
      - if any NaN/Inf appears in (a-b), treat as +inf so it can't hide.
    """
    m = 0.0
    for k in a.keys():
        diff = (a[k] - b[k]).abs()
        d = torch.nan_to_num(diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max().item()
        if d > m:
            m = float(d)
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
def assert_params_equal_allclose(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    *,
    tag: str,
    atol: float,
    rtol: float,
):
    """
    항상 allclose로만 비교한다.
    strict(equal) 비교는 아예 하지 않는다.
    """
    ka = set(a.keys())
    kb = set(b.keys())
    if ka != kb:
        missing = sorted(list(ka - kb))
        extra = sorted(list(kb - ka))
        fail(f"[{tag}] key mismatch. missing_in_b={missing}, extra_in_b={extra}")

    keys = sorted(list(ka))

    # meta check + maxdiff (NaN-safe)
    maxdiff = 0.0
    for k in keys:
        ta = a[k]
        tb = b[k]
        if ta.shape != tb.shape or ta.dtype != tb.dtype or ta.device != tb.device:
            fail(
                f"[{tag}] meta mismatch for '{k}': "
                f"a(shape={tuple(ta.shape)},dtype={ta.dtype},dev={ta.device}) vs "
                f"b(shape={tuple(tb.shape)},dtype={tb.dtype},dev={tb.device})"
            )
        diff = (ta - tb).abs()
        d = torch.nan_to_num(diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max().item()
        if d > maxdiff:
            maxdiff = float(d)

    # non-finite guard
    def _has_nonfinite(snap: dict[str, torch.Tensor]) -> tuple[bool, list[tuple[str, int, int]]]:
        bad = []
        for k in keys:
            _, nn, ni = _tensor_nan_inf_stats(snap[k])
            if nn or ni:
                bad.append((k, nn, ni))
        return (len(bad) > 0), bad

    ha, bada = _has_nonfinite(a)
    hb, badb = _has_nonfinite(b)
    if ha or hb:
        la = "\n".join([f"  - {k}: nan={nn}, inf={ni}" for (k, nn, ni) in bada[:16]])
        lb = "\n".join([f"  - {k}: nan={nn}, inf={ni}" for (k, nn, ni) in badb[:16]])
        fail(
            f"[{tag}] non-finite present, cannot compare.\n"
            f"[A non-finite]\n{la if la else '  (none)'}\n"
            f"[B non-finite]\n{lb if lb else '  (none)'}\n"
            f"max_abs_diff(all, nan-safe)={maxdiff:.6e}"
        )

    # allclose
    bad = []
    for k in keys:
        if not torch.allclose(a[k], b[k], atol=atol, rtol=rtol):
            bad.append(k)

    if bad:
        k0 = bad[0]
        diff0 = (a[k0] - b[k0]).abs()
        md0 = float(torch.nan_to_num(diff0, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max().item())
        fail(
            f"[{tag}] params not equal (allclose). first_bad='{k0}', "
            f"max_abs_diff(first_bad, nan-safe)={md0:.6e}, "
            f"max_abs_diff(all, nan-safe)={maxdiff:.6e}, atol={atol}, rtol={rtol}, bad_count={len(bad)}"
        )

    ok(f"[{tag}] params equal (allclose). max_abs_diff(all, nan-safe)={maxdiff:.6e}, atol={atol}, rtol={rtol}")


@torch.no_grad()
def max_param_stepdiff(model, snap: dict[str, torch.Tensor]) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        diff = (p.data - snap[n]).abs()
        d = torch.nan_to_num(diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max().item()
        if d > m:
            m = float(d)
    return m


def main():
    print("=== PR6.1 verify: replay == IRExecutor (same initial state) + determinism + adam_input_check ===")
    print(
        f"seed={SEED}, warmup_runs={WARMUP_RUNS}, replay_n={REPLAY_N}, ir_exec_runs={IR_EXEC_RUNS}, "
        f"eq(allclose atol={EQ_ATOL}, rtol={EQ_RTOL}), debug_nan={DEBUG_NAN}"
    )

    if not torch.cuda.is_available():
        fail("CUDA not available")

    seed_all(SEED)
    torch.set_grad_enabled(False)

    # AICF imports (UPDATED for new core structure)
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

    # enforce adam_step dispatch in runtime
    os.environ.setdefault("AICF_ENFORCE_ADAMSTEP_RUNTIME", "1")

    # compile+capture
    # NOTE: attach_env=True ensures IRExecutor.from_artifact works.
    art = model.compile(
        optim=optim,
        x=x,
        t=t,
        loss="mse",
        name="train_step_pr6_1",
        warmup_runs=WARMUP_RUNS,
        warmup_sync=True,
        validate=True,
        trace=True,
        enforce_ops=("adam_step",),
        torch_sync=True,
        attach_env=True,
    )

    art.assert_runtime_matches_lowering(model, trace_filter=True)
    ok("[lowering/trace] forward-slice + optim-slice match")

    exe = IRExecutor.from_artifact(art)

    # ------------------------------------------------------------
    # A) replay_once mutates
    # ------------------------------------------------------------
    st0 = TrainState.capture(model, optim)
    snap0 = snapshot_named_params(model)

    art.backend.replay()
    torch.cuda.synchronize()

    st1 = TrainState.capture(model, optim)
    snap1 = snapshot_named_params(model)

    assert_params_finite(snap1, tag="after_replay_once")

    if int(st1.step.item()) == int(st0.step.item()):
        fail("replay did not advance step")
    assert_params_changed(snap0, snap1, tag="replay_once")
    ok(f"[replay_once] step {int(st0.step.item())} -> {int(st1.step.item())}")

    # ------------------------------------------------------------
    # B) irexec_once mutates (restore first)
    # ------------------------------------------------------------
    st0.restore(model, optim)
    torch.cuda.synchronize()

    snap2 = snapshot_named_params(model)
    for _ in range(IR_EXEC_RUNS):
        exe.run(debug_nan=bool(DEBUG_NAN))

    torch.cuda.synchronize()
    snap3 = snapshot_named_params(model)

    assert_params_finite(snap3, tag="after_irexec_once")
    assert_params_changed(snap2, snap3, tag="ir_exec")
    ok(f"[ir_exec] runs={IR_EXEC_RUNS}")

    # ------------------------------------------------------------
    # C) same initial state: replay 1 step ~= irexec 1 step (allclose)
    # ------------------------------------------------------------
    base = TrainState.capture(model, optim)

    # replay path
    base.restore(model, optim)
    torch.cuda.synchronize()
    snap_r0 = snapshot_named_params(model)
    art.backend.replay()
    torch.cuda.synchronize()
    snap_r1 = snapshot_named_params(model)
    assert_params_finite(snap_r1, tag="after_eq_replay")
    assert_params_changed(snap_r0, snap_r1, tag="eq_replay")

    # irexec path
    base.restore(model, optim)
    torch.cuda.synchronize()
    snap_i0 = snapshot_named_params(model)
    exe.run(debug_nan=bool(DEBUG_NAN))
    torch.cuda.synchronize()
    snap_i1 = snapshot_named_params(model)
    assert_params_finite(snap_i1, tag="after_eq_irexec")
    assert_params_changed(snap_i0, snap_i1, tag="eq_irexec")

    # compare final params (ALLCLOSE ONLY)
    assert_params_equal_allclose(snap_r1, snap_i1, tag="replay_vs_irexec_equal", atol=EQ_ATOL, rtol=EQ_RTOL)

    # ------------------------------------------------------------
    # D) Determinism: replay stepdiff sequence matches after restore
    # ------------------------------------------------------------
    base.restore(model, optim)
    torch.cuda.synchronize()

    A: list[float] = []
    snaps = snapshot_named_params(model)
    for i in range(REPLAY_N):
        art.backend.replay()
        torch.cuda.synchronize()
        cur = snapshot_named_params(model)
        assert_params_finite(cur, tag=f"det_A_nonfinite@{i:02d}")
        A.append(max_param_stepdiff(model, snaps))
        snaps = cur
        if PRINT_EVERY and (i % PRINT_EVERY == 0):
            print(f"[A] i={i:02d} stepdiff={A[-1]:.6e}")

    base.restore(model, optim)
    torch.cuda.synchronize()

    B: list[float] = []
    snaps = snapshot_named_params(model)
    for i in range(REPLAY_N):
        art.backend.replay()
        torch.cuda.synchronize()
        cur = snapshot_named_params(model)
        assert_params_finite(cur, tag=f"det_B_nonfinite@{i:02d}")
        B.append(max_param_stepdiff(model, snaps))
        snaps = cur
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
