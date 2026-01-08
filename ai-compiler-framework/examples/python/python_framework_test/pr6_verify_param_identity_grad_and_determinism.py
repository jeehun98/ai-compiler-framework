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
SEED = int(os.environ.get("AICF_SEED", "0"))
WARMUP_RUNS = int(os.environ.get("AICF_WARMUP_RUNS", "2"))
TRACE_DUMP = int(os.environ.get("AICF_TRACE_DUMP", "1"))

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
    return {n: p.data.detach().clone() for n, p in model.named_parameters()}

@torch.no_grad()
def max_param_diff(model, snaps) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - snaps[n]).abs().max().item()
        if d > m:
            m = float(d)
    return m

def _ptr(x: torch.Tensor) -> int:
    return int(x.data_ptr())

def dump_param_grad_state(model, optim, tag: str):
    print(f"--- [dump] {tag} ---")
    name_to_p = {n: p for n, p in model.named_parameters()}
    for i, p in enumerate(optim.params):
        # find name (best-effort)
        nm = None
        for n, mp in name_to_p.items():
            if mp is p:
                nm = n
                break
        nm = nm or f"optim.param[{i}]"

        g = p.grad
        if g is None:
            print(f"[dump] {nm}: grad=None  (p.data_ptr={_ptr(p.data)})")
            continue

        # grad can be aicf_fw.core.tensor.Tensor or torch.Tensor (depending on your codepaths)
        g_data = getattr(g, "data", None)
        g_type = type(g).__name__
        gdata_type = type(g_data).__name__ if g_data is not None else "None"

        if isinstance(g_data, torch.Tensor):
            print(f"[dump] {nm}: grad={g_type}, grad.data={gdata_type}, grad.data_ptr={_ptr(g_data)}")
        else:
            # could be None or unexpected type
            print(f"[dump] {nm}: grad={g_type}, grad.data={gdata_type} (NON-TORCH or None)")

def check_param_identity_by_object(model, optim):
    # strongest: object identity match (not just data_ptr)
    model_ps = [p for _, p in model.named_parameters()]
    if len(model_ps) != len(optim.params):
        fail(f"[param-id] len mismatch: model={len(model_ps)} optim={len(optim.params)}")

    bad = []
    for i, (mp, op) in enumerate(zip(model_ps, optim.params)):
        if mp is not op:
            bad.append(i)

    if bad:
        warn(f"[param-id] object identity mismatch at indices={bad} (can still work, but grads may not be shared)")
    else:
        ok("[param-id] optimizer params are identical objects to model params")

def check_param_identity_by_data_ptr(model, optim):
    m_ptrs = {n: _ptr(p.data) for n, p in model.named_parameters()}
    o_ptrs = [_ptr(p.data) for p in optim.params]
    if set(m_ptrs.values()) != set(o_ptrs):
        fail("[param-id] optimizer params != model params (data_ptr mismatch)")
    ok("[param-id] optimizer params match model params (by data_ptr)")

def check_all_grads_non_none(optim, tag: str):
    bad = []
    for i, p in enumerate(optim.params):
        if p.grad is None:
            bad.append(i)
    if bad:
        fail(f"[grad] {tag}: grad is None for indices={bad}")
    ok(f"[grad] {tag}: all grads are non-None ({len(optim.params)} params)")

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=== PR6 verify: adam_step invocation + capture/replay mutation ===")
    print(f"seed={SEED}, warmup_runs={WARMUP_RUNS}")

    if not torch.cuda.is_available():
        fail("CUDA not available")

    seed_all(SEED)

    # AICF imports
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
    ok(f"Backend set: {type(bk).__name__}")

    backend.capture_reset()
    torch.cuda.synchronize()

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

    check_param_identity_by_data_ptr(model, optim)
    check_param_identity_by_object(model, optim)

    # --------------------------------------------------------
    # Count calls in TWO places:
    #  1) backend.op_call_out("adam_step", ...)  (runtime dispatch)
    #  2) F.adam_step_(...) function call itself (python level)
    # --------------------------------------------------------
    counts = {"backend_adam": 0, "func_adam": 0}

    # (1) backend op_call_out wrapper
    orig_op_call_out = backend.op_call_out
    def op_call_out_count(op: str, inputs, outputs, attrs):
        op_l = op.strip().lower().replace("-", "_")
        if op_l in ("adam_step", "adamstep"):
            counts["backend_adam"] += 1
        return orig_op_call_out(op, inputs, outputs, attrs)
    backend.op_call_out = op_call_out_count  # type: ignore

    # (2) functional.adam_step_ wrapper (this is the most important)
    orig_F_adam_step_ = F.adam_step_
    def F_adam_step__count(*args, **kwargs):
        counts["func_adam"] += 1
        return orig_F_adam_step_(*args, **kwargs)
    F.adam_step_ = F_adam_step__count  # type: ignore

    # --------------------------------------------------------
    # Train step
    # --------------------------------------------------------
    def train_step():
        optim.zero_grad()
        y = model(x)
        dY = F.mse_grad(y, t)
        autograd_backward(y, grad=dY, accumulate=False)

        # grads must exist if adam_step should run
        dump_param_grad_state(model, optim, tag="after-backward (pre-step)")
        check_all_grads_non_none(optim, tag="after-backward (pre-step)")

        optim.step_()

    # --------------------------------------------------------
    # Warmup: materialize buffers + make sure step path works
    # --------------------------------------------------------
    warmup_capture_safe(train_step=train_step, runs=WARMUP_RUNS, sync=True)
    print("[warmup] functional buffers =", functional_buffer_stats())

    # after warmup, we expect adam_step to have been called at least once
    if counts["func_adam"] == 0:
        fail("[warmup] F.adam_step_ was never called even during warmup. Optimizer.step_ is skipping adam_step.")
    ok(f"[warmup] counts: func_adam={counts['func_adam']} backend_adam={counts['backend_adam']}")

    # reset counters for capture test
    counts["func_adam"] = 0
    counts["backend_adam"] = 0

    backend.capture_reset()
    torch.cuda.synchronize()

    # --------------------------------------------------------
    # Capture + trace
    # --------------------------------------------------------
    backend.trace_reset()
    backend.trace_enable(True)

    snap_pre_cap = snapshot_params(model)

    backend.capture_begin()
    train_step()
    backend.capture_end()
    torch.cuda.synchronize()

    diff_cap = max_param_diff(model, snap_pre_cap)
    ok(f"[capture-run] max_param_diff={diff_cap:.6e} func_adam={counts['func_adam']} backend_adam={counts['backend_adam']}")

    trace_ops = backend.trace_get()
    backend.trace_enable(False)

    if TRACE_DUMP:
        print("=== TRACE OPS (runtime) ===")
        for i, op in enumerate(trace_ops):
            print(f"[trace {i:02d}] op={op}")

    # --------------------------------------------------------
    # Replay mutation check
    # --------------------------------------------------------
    snap_pre_replay = snapshot_params(model)
    backend.replay()
    torch.cuda.synchronize()
    diff_replay = max_param_diff(model, snap_pre_replay)
    ok(f"[replay] max_param_diff={diff_replay:.6e}")

    # --------------------------------------------------------
    # Diagnose (hard, unambiguous)
    # --------------------------------------------------------
    if counts["func_adam"] == 0:
        fail(
            "[diagnose] F.adam_step_ was not called inside capture-run. "
            "This means Optimizer.step_ skipped the adam loop (most likely p.grad==None inside step_). "
            "Check the [dump] output right before optim.step_()."
        )

    if counts["backend_adam"] == 0:
        fail(
            "[diagnose] F.adam_step_ was called but backend.op_call_out('adam_step') was not. "
            "This implies you hit tracing path inside F.adam_step_ (is_tracing True) or early-returned."
        )

    if diff_cap == 0.0:
        fail(
            "[diagnose] AdamStep was dispatched during capture-run but params did not change. "
            "This points to kernel/dispatch/output-binding bug (adam_step not mutating p)."
        )

    if diff_replay == 0.0:
        fail(
            "[diagnose] Params changed during capture-run but not during replay. "
            "This means AdamStep executed outside captured graph (stream/capture boundary issue)."
        )

    ok("[diagnose] OK: adam_step invoked + captured + replay mutates params")
    ok("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
