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

SEED = int(os.environ.get("AICF_SEED", "0"))
WARMUP_RUNS = int(os.environ.get("AICF_WARMUP_RUNS", "2"))
TRACE_DUMP = int(os.environ.get("AICF_TRACE_DUMP", "1"))

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ok(msg: str): print(f"[OK] {msg}")
def warn(msg: str): print(f"[WARN] {msg}")
def fail(msg: str, code: int = 1):
    print(f"[FAIL] {msg}")
    raise SystemExit(code)

def _ptr(x: torch.Tensor) -> int:
    return int(x.data_ptr())

@torch.no_grad()
def snapshot_params(model):
    return {n: p.data.detach().clone() for n, p in model.named_parameters()}

@torch.no_grad()
def max_param_diff(model, snaps) -> float:
    m = 0.0
    for n, p in model.named_parameters():
        d = (p.data - snaps[n]).abs().max().item()
        if d > m: m = float(d)
    return m

def dump_module_identities(tag: str, trace_mod, func_mod):
    print(f"--- [mods] {tag} ---")
    print(f"[mods] trace.__file__ = {getattr(trace_mod, '__file__', None)}")
    print(f"[mods] functional.__file__ = {getattr(func_mod, '__file__', None)}")
    print(f"[mods] id(trace_mod)={id(trace_mod)} id(func_mod)={id(func_mod)}")

def dump_param_grads(model, optim, tag: str):
    print(f"--- [dump] {tag} ---")
    name_to_p = {n: p for n, p in model.named_parameters()}
    for i, p in enumerate(optim.params):
        nm = None
        for n, mp in name_to_p.items():
            if mp is p:
                nm = n
                break
        nm = nm or f"optim.param[{i}]"
        g = p.grad
        if g is None:
            print(f"[dump] {nm}: grad=None (p.ptr={_ptr(p.data)})")
        else:
            g_data = getattr(g, "data", None)
            # note: torch.Tensor type name is also "Tensor"
            print(
                f"[dump] {nm}: grad_type={type(g).__name__} "
                f"grad.data_type={type(g_data).__name__} "
                f"(p.ptr={_ptr(p.data)})"
            )

def main():
    print("=== PR7 verify: adam_step path (tracing/dup modules) ===")
    print(f"seed={SEED}, warmup_runs={WARMUP_RUNS}")

    if not torch.cuda.is_available():
        fail("CUDA not available")

    seed_all(SEED)

    # imports
    from aicf_fw.backend.aicf_backend import AICFBackend
    from aicf_fw.backend import set_backend
    from aicf_fw.core.tensor import Tensor
    from aicf_fw.core.autograd import backward as autograd_backward
    from aicf_fw.core.warmup import warmup_capture_safe
    from aicf_fw.core.functional import functional_buffer_stats
    from aicf_fw.nn.linear import Linear
    from aicf_fw.nn.relu import ReLU
    from aicf_fw.nn.sequential import Sequential
    from aicf_fw.optim.adam import Adam

    # IMPORTANT: import the exact modules Adam uses
    import aicf_fw.core.trace as trace_mod
    import aicf_fw.core.functional as func_mod

    backend = AICFBackend()
    set_backend(backend)
    ok(f"Backend set: {type(backend).__name__}")

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

    dump_module_identities("initial", trace_mod, func_mod)
    ok(f"[trace] is_tracing() initially = {trace_mod.is_tracing()}")

    # counters
    counts = {"func_adam": 0, "backend_adam": 0}

    # patch backend.op_call_out to count adam_step, AND to log ops during step_ only
    orig_op_call_out = backend.op_call_out
    def op_call_out_count(op: str, inputs, outputs, attrs):
        op_l = op.strip().lower().replace("-", "_")
        if op_l == "adam_step":
            counts["backend_adam"] += 1
        return orig_op_call_out(op, inputs, outputs, attrs)
    backend.op_call_out = op_call_out_count  # type: ignore

    # patch functional.adam_step_ to print is_tracing at call-time
    orig_adam_step = func_mod.adam_step_
    def adam_step_dbg(*args, **kwargs):
        counts["func_adam"] += 1
        it = trace_mod.is_tracing()
        if it:
            warn("[dbg] is_tracing()==True inside adam_step_ (this explains backend_adam=0)")
        return orig_adam_step(*args, **kwargs)
    func_mod.adam_step_ = adam_step_dbg  # type: ignore

    # one step
    def train_step():
        # sanity: detect tracing leak
        if trace_mod.is_tracing():
            fail("[sanity] tracing is ON during runtime train_step()")

        optim.zero_grad()
        y = model(x)
        dY = func_mod.mse_grad(y, t)
        autograd_backward(y, grad=dY, accumulate=False)

        dump_param_grads(model, optim, "after-backward (pre-step)")
        # isolate trace during optimizer step only
        backend.trace_reset()
        optim.step_()
        step_ops = backend.trace_get()
        return step_ops

    # warmup
    warmup_capture_safe(train_step=train_step, runs=WARMUP_RUNS, sync=True)
    print("[warmup] functional buffers =", functional_buffer_stats())
    ok(f"[warmup] counts: func_adam={counts['func_adam']} backend_adam={counts['backend_adam']}")

    # reset counts
    counts["func_adam"] = 0
    counts["backend_adam"] = 0

    backend.capture_reset()
    torch.cuda.synchronize()

    # capture
    backend.trace_reset()
    backend.trace_enable(True)

    snap_pre = snapshot_params(model)

    backend.capture_begin()
    step_ops = train_step()
    backend.capture_end()
    torch.cuda.synchronize()

    diff_cap = max_param_diff(model, snap_pre)
    ok(f"[capture-run] max_param_diff={diff_cap:.6e} func_adam={counts['func_adam']} backend_adam={counts['backend_adam']}")

    if TRACE_DUMP:
        print("=== TRACE OPS (during optim.step_ only) ===")
        for i, op in enumerate(step_ops):
            print(f"[step-op {i:02d}] {op}")

    # replay mutation
    snap_pre_rep = snapshot_params(model)
    backend.replay()
    torch.cuda.synchronize()
    diff_rep = max_param_diff(model, snap_pre_rep)
    ok(f"[replay] max_param_diff={diff_rep:.6e}")

    # diagnose
    if counts["func_adam"] == 0:
        fail("[diagnose] adam_step_ not called at all (optimizer skipped loop)")
    if counts["backend_adam"] == 0:
        fail(
            "[diagnose] adam_step_ called but backend op_call_out('adam_step') never happened.\n"
            "Most likely is_tracing() is unexpectedly True inside adam_step_, or you have duplicated module instances.\n"
            "Check [mods] paths + [dbg] warning above."
        )
    if diff_cap == 0.0:
        fail("[diagnose] adam_step dispatched but params didn't change (kernel/dispatch bug)")
    if diff_rep == 0.0:
        fail("[diagnose] params changed in capture-run but not replay (not captured / stream boundary issue)")

    ok("OK")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
