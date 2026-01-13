from __future__ import annotations

import os
import sys
from pathlib import Path
import random
import numpy as np
import torch
import zlib
import inspect
from dataclasses import dataclass
from typing import Any


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

EQ_ATOL = float(os.environ.get("AICF_EQ_ATOL", "1e-4"))
EQ_RTOL = float(os.environ.get("AICF_EQ_RTOL", "1e-4"))

DEBUG_NAN = int(os.environ.get("AICF_DEBUG_NAN", "1"))  # 1=on, 0=off
DEBUG_PTR = int(os.environ.get("AICF_DEBUG_PTR", "1"))  # 1=print param ptrs
DEBUG_OPT_STATE = int(os.environ.get("AICF_DEBUG_OPT_STATE", "1"))  # 1=compare opt state too
DEBUG_GRADS = int(os.environ.get("AICF_DEBUG_GRADS", "1"))  # 1=compare grads too

DEBUG_INTERMEDIATE = int(os.environ.get("AICF_DEBUG_INTERMEDIATE", "0"))  # 1=on
DIVERGE_LIMIT = int(os.environ.get("AICF_IREXEC_DIVERGE_LIMIT", "50"))

DEBUG_ADAM_INPUTS = int(os.environ.get("AICF_DEBUG_ADAM_INPUTS", "0"))  # 1=on


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


def _call_run_safely(exe, **kwargs):
    """
    exe.run()이 지원하는 kwarg만 필터링해서 호출.
    IRExecutor.run 시그니처가 바뀌어도 테스트가 안 깨지게 함.
    """
    run_fn = getattr(exe, "run")
    sig = inspect.signature(run_fn)
    supported = sig.parameters.keys()
    filtered = {k: v for k, v in kwargs.items() if k in supported}
    return run_fn(**filtered)


# ----------------------------
# AICF Tensor/grad helpers
# ----------------------------
def _unwrap_torch_tensor(x: Any) -> torch.Tensor | None:
    """
    x가 torch.Tensor면 그대로.
    aicf_fw.core.autograd.Tensor 같은 래퍼면 내부 torch.Tensor(.data or .t)를 최대한 찾아서 반환.
    """
    if x is None:
        return None
    if isinstance(x, torch.Tensor):
        return x
    if hasattr(x, "data") and isinstance(getattr(x, "data"), torch.Tensor):
        return getattr(x, "data")
    if hasattr(x, "t") and isinstance(getattr(x, "t"), torch.Tensor):
        return getattr(x, "t")
    return None


@torch.no_grad()
def snapshot_named_params(model) -> dict[str, torch.Tensor]:
    return {n: p.data.detach().clone() for n, p in model.named_parameters()}


@torch.no_grad()
def snapshot_named_grads(model) -> dict[str, torch.Tensor]:
    """
    grad 스냅샷:
      - p.grad가 torch.Tensor가 아닐 수 있음(aicf Tensor wrapper)
      - 없으면 0 텐서로 대체
    """
    out: dict[str, torch.Tensor] = {}
    for n, p in model.named_parameters():
        g_raw = getattr(p, "grad", None)
        g_t = _unwrap_torch_tensor(g_raw)

        if g_t is None:
            g_t = p.data.new_zeros(p.data.shape)
        else:
            g_t = g_t.detach()

        out[n] = g_t.clone()
    return out


@torch.no_grad()
def snapshot_opt_state(optim) -> dict[str, torch.Tensor]:
    """
    Optimizer state snapshot:
      - step, bc1_inv, bc2_inv
      - m[i], v[i]
    """
    out: dict[str, torch.Tensor] = {}
    out["step"] = optim.step.detach().clone()
    out["bc1_inv"] = optim.bc1_inv.detach().clone()
    out["bc2_inv"] = optim.bc2_inv.detach().clone()

    for i in sorted(list(optim.m.keys())):
        out[f"m[{i}]"] = optim.m[i].data.detach().clone()
        out[f"v[{i}]"] = optim.v[i].data.detach().clone()
    return out


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
        fail(f"[{tag}] non-finite detected:\n{lines}")


@torch.no_grad()
def assert_params_changed(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor], *, tag: str):
    changed = 0
    for n, b in before.items():
        a = after[n]
        if not torch.equal(b, a):
            changed += 1
    if changed == 0:
        fail(f"[{tag}] params did NOT change")
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
    ka = set(a.keys())
    kb = set(b.keys())
    if ka != kb:
        missing = sorted(list(ka - kb))
        extra = sorted(list(kb - ka))
        fail(f"[{tag}] key mismatch. missing_in_b={missing}, extra_in_b={extra}")

    keys = sorted(list(ka))
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

    bad = []
    for k in keys:
        if not torch.allclose(a[k], b[k], atol=atol, rtol=rtol):
            bad.append(k)

    if bad:
        print(f"[DBG] bad keys ({len(bad)}): {bad}")
        k0 = bad[0]
        diff0 = (a[k0] - b[k0]).abs()
        md0 = float(torch.nan_to_num(diff0, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max().item())
        fail(
            f"[{tag}] not equal (allclose). first_bad='{k0}', "
            f"max_abs_diff(first_bad, nan-safe)={md0:.6e}, "
            f"max_abs_diff(all, nan-safe)={maxdiff:.6e}, atol={atol}, rtol={rtol}, bad_count={len(bad)}"
        )

    ok(f"[{tag}] equal (allclose). max_abs_diff(all, nan-safe)={maxdiff:.6e}, atol={atol}, rtol={rtol}")


@torch.no_grad()
def check_allclose_or_return_bad(
    a: dict[str, torch.Tensor],
    b: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
) -> list[str]:
    keys = sorted(list(a.keys()))
    bad = []
    for k in keys:
        if not torch.allclose(a[k], b[k], atol=atol, rtol=rtol):
            bad.append(k)
    return bad


@torch.no_grad()
def max_param_stepdiff(model, snap: dict[str, torch.Tensor]) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        diff = (p.data - snap[n]).abs()
        d = torch.nan_to_num(diff, nan=float("inf"), posinf=float("inf"), neginf=float("inf")).max().item()
        if d > m:
            m = float(d)
    return m


@torch.no_grad()
def print_param_ptrs(model, *, header: str):
    print(f"--- {header} param ptrs ---")
    for n, p in model.named_parameters():
        t = p.data
        t32 = t.float()
        mean = float(t32.mean().item())
        mx = float(t32.max().item())
        print(f"[P] {n:12s} ptr={t.data_ptr():>12d} mean={mean:+.6e} max={mx:+.6e}")


@torch.no_grad()
def print_opt_state_ptrs(optim, *, header: str):
    print(f"--- {header} opt_state ptrs ---")
    for name in ["step", "bc1_inv", "bc2_inv"]:
        t = getattr(optim, name).detach()
        val = float(t.item()) if t.numel() == 1 else float(t.float().mean().item())
        print(f"[O] {name:8s} ptr={t.data_ptr():>12d} val={val:+.6e} shape={tuple(t.shape)}")
    for i in sorted(list(optim.m.keys())):
        mt = optim.m[i].data.detach()
        vt = optim.v[i].data.detach()
        mm = float(mt.float().mean().item()) if mt.numel() else 0.0
        vm = float(vt.float().mean().item()) if vt.numel() else 0.0
        print(
            f"[O] m[{i}] ptr={mt.data_ptr():>12d} mean={mm:+.6e} shape={tuple(mt.shape)} | "
            f"v[{i}] ptr={vt.data_ptr():>12d} mean={vm:+.6e} shape={tuple(vt.shape)}"
        )


# ------------------------------------------------------------
# Optional: IRExecutor sig dump (only if patched runtime.py supports get_last_sigs)
# ------------------------------------------------------------
def dump_irexec_sig_for_keys(exe, keys: list[str], *, limit: int = 200):
    if not hasattr(exe, "get_last_sigs"):
        warn("IRExecutor has no get_last_sigs()")
        return
    sigs = exe.get_last_sigs()
    if not isinstance(sigs, dict) or not sigs:
        warn("IRExecutor sigs empty; set DEBUG_INTERMEDIATE=1 to collect")
        return

    hits: list[tuple[str, Any]] = []

    if DEBUG_ADAM_INPUTS:
        for k, sig in sigs.items():
            lk = k.lower()
            if (":adam_step:" in lk) and (":adam_in" in lk):
                hits.append((k, sig))
        if hits:
            print(f"[DBG] irexec adam_step input sigs: {len(hits)}")
            for k, sig in hits[:limit]:
                print(f"  - {k} -> {sig}")
            return

    targets = [k.lower() for k in keys]
    for k, sig in sigs.items():
        lk = k.lower()
        if (":out" in lk) and lk.endswith(":after") and any(t in lk for t in targets):
            hits.append((k, sig))

    if not hits:
        outs = [(k, sig) for k, sig in sigs.items() if (":out" in k.lower()) and k.lower().endswith(":after")]
        hits = outs[-min(len(outs), limit):]

    print(f"[DBG] irexec sig hits: {len(hits)} (show up to {limit})")
    for k, sig in hits[:limit]:
        print(f"  - {k} -> {sig}")


def main():
    print("=== PR6.1 verify: replay == IRExecutor (env_provider) + determinism ===")
    print(
        f"seed={SEED}, warmup_runs={WARMUP_RUNS}, replay_n={REPLAY_N}, ir_exec_runs={IR_EXEC_RUNS}, "
        f"eq(allclose atol={EQ_ATOL}, rtol={EQ_RTOL}), debug_nan={DEBUG_NAN}, "
        f"debug_ptr={DEBUG_PTR}, debug_opt_state={DEBUG_OPT_STATE}, debug_grads={DEBUG_GRADS}, "
        f"debug_intermediate={DEBUG_INTERMEDIATE}, diverge_limit={DIVERGE_LIMIT}, debug_adam_inputs={DEBUG_ADAM_INPUTS}"
    )

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

    backend = AICFBackend()
    set_backend(backend)
    bk = get_backend()
    ok(f"Backend set: {type(bk).__name__}")

    backend.capture_reset()
    torch.cuda.synchronize()

    model = Sequential(
        Linear(8, 8, device="cuda", dtype=torch.float32),
        ReLU(),
        Linear(8, 8, device="cuda", dtype=torch.float32),
    )

    # 기본값 1e-3 (0이면 params changed 검사에서 실패 가능)
    LR = float(os.environ.get("AICF_LR", "1e-3"))
    optim = Adam(model, lr=LR, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=None)

    x = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False, name="x")
    t = Tensor(torch.randn(64, 8, device="cuda", dtype=torch.float32), requires_grad=False, name="t")

    for n, p in list(model.named_parameters()):
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    os.environ.setdefault("AICF_ENFORCE_ADAMSTEP_RUNTIME", "1")

    # compile()가 art.env_provider를 붙여주게 되어있다는 가정 (네가 수정한 module.py 기준)
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
        attach_env=True,  # env_provider attached
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
        _call_run_safely(exe, debug_nan=bool(DEBUG_NAN), debug_intermediate=bool(DEBUG_INTERMEDIATE), return_intermediates=False)
    torch.cuda.synchronize()
    snap3 = snapshot_named_params(model)

    assert_params_finite(snap3, tag="after_irexec_once")
    assert_params_changed(snap2, snap3, tag="ir_exec")
    ok(f"[ir_exec] runs={IR_EXEC_RUNS}")

    # ------------------------------------------------------------
    # C) same initial state: replay 1 step ~= irexec 1 step (allclose)
    # ------------------------------------------------------------
    base = TrainState.capture(model, optim)

    # restore stability checks (두 번 restore해도 동일해야 함)
    base.restore(model, optim)
    torch.cuda.synchronize()
    sA0 = snapshot_named_params(model)
    oA0 = snapshot_opt_state(optim)
    gA0 = snapshot_named_grads(model) if DEBUG_GRADS else None

    base.restore(model, optim)
    torch.cuda.synchronize()
    sA1 = snapshot_named_params(model)
    oA1 = snapshot_opt_state(optim)
    gA1 = snapshot_named_grads(model) if DEBUG_GRADS else None

    assert_params_equal_allclose(sA0, sA1, tag="start_state_stable_after_restore(params)", atol=0.0, rtol=0.0)
    ok("[start_state] restore stability: params OK")

    if DEBUG_OPT_STATE:
        assert_params_equal_allclose(oA0, oA1, tag="start_state_stable_after_restore(opt_state)", atol=0.0, rtol=0.0)
        ok("[start_state] restore stability: opt-state OK")

    if DEBUG_GRADS:
        assert_params_equal_allclose(gA0, gA1, tag="start_state_stable_after_restore(grads)", atol=0.0, rtol=0.0)
        ok("[start_state] restore stability: grads OK")

    # -------- replay path --------
    base.restore(model, optim)
    torch.cuda.synchronize()

    snap_r0 = snapshot_named_params(model)
    opt_r0 = snapshot_opt_state(optim) if DEBUG_OPT_STATE else None
    grd_r0 = snapshot_named_grads(model) if DEBUG_GRADS else None

    if DEBUG_PTR:
        print_param_ptrs(model, header="start_replay")
        if DEBUG_OPT_STATE:
            print_opt_state_ptrs(optim, header="start_replay")

    art.backend.replay()
    torch.cuda.synchronize()

    snap_r1 = snapshot_named_params(model)
    opt_r1 = snapshot_opt_state(optim) if DEBUG_OPT_STATE else None
    grd_r1 = snapshot_named_grads(model) if DEBUG_GRADS else None

    assert_params_finite(snap_r1, tag="after_eq_replay")
    assert_params_changed(snap_r0, snap_r1, tag="eq_replay")

    # -------- irexec path --------
    base.restore(model, optim)
    torch.cuda.synchronize()

    snap_i0 = snapshot_named_params(model)
    opt_i0 = snapshot_opt_state(optim) if DEBUG_OPT_STATE else None
    grd_i0 = snapshot_named_grads(model) if DEBUG_GRADS else None

    if DEBUG_PTR:
        print_param_ptrs(model, header="start_irexec")
        if DEBUG_OPT_STATE:
            print_opt_state_ptrs(optim, header="start_irexec")

    assert_params_equal_allclose(snap_r0, snap_i0, tag="start_state_equal_between_paths(params)", atol=0.0, rtol=0.0)
    if DEBUG_OPT_STATE:
        assert_params_equal_allclose(opt_r0, opt_i0, tag="start_state_equal_between_paths(opt_state)", atol=0.0, rtol=0.0)
    if DEBUG_GRADS:
        assert_params_equal_allclose(grd_r0, grd_i0, tag="start_state_equal_between_paths(grads)", atol=0.0, rtol=0.0)

    _call_run_safely(exe, debug_nan=bool(DEBUG_NAN), debug_intermediate=bool(DEBUG_INTERMEDIATE), return_intermediates=False)
    torch.cuda.synchronize()

    snap_i1 = snapshot_named_params(model)
    opt_i1 = snapshot_opt_state(optim) if DEBUG_OPT_STATE else None
    grd_i1 = snapshot_named_grads(model) if DEBUG_GRADS else None

    assert_params_finite(snap_i1, tag="after_eq_irexec")
    assert_params_changed(snap_i0, snap_i1, tag="eq_irexec")

    # final compare
    assert_params_equal_allclose(snap_r1, snap_i1, tag="replay_vs_irexec_equal(params)", atol=EQ_ATOL, rtol=EQ_RTOL)

    if DEBUG_OPT_STATE:
        bad_opt = check_allclose_or_return_bad(opt_r1, opt_i1, atol=EQ_ATOL, rtol=EQ_RTOL)
        if bad_opt:
            print(f"[DBG] opt_state bad keys: {bad_opt}")
            print_opt_state_ptrs(optim, header="(current optim) at opt_state mismatch")
            dump_irexec_sig_for_keys(exe, bad_opt, limit=200)
        assert_params_equal_allclose(opt_r1, opt_i1, tag="replay_vs_irexec_equal(opt_state)", atol=EQ_ATOL, rtol=EQ_RTOL)

    if DEBUG_GRADS:
        bad_g = check_allclose_or_return_bad(grd_r1, grd_i1, atol=EQ_ATOL, rtol=EQ_RTOL)
        if bad_g:
            print(f"[DBG] grads bad keys: {bad_g}")
            dump_irexec_sig_for_keys(exe, bad_g, limit=200)
        assert_params_equal_allclose(grd_r1, grd_i1, tag="replay_vs_irexec_equal(grads)", atol=EQ_ATOL, rtol=EQ_RTOL)

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
