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
TRACE_FILTER = int(os.environ.get("AICF_TRACE_FILTER", "1"))

# DO NOT run AICF forward in the determinism loop (it mutates pool)
PRINT_LOSS_EVERY = int(os.environ.get("AICF_PRINT_LOSS_EVERY", "0"))  # 0=never

CHECK_RESTORE = int(os.environ.get("AICF_CHECK_RESTORE", "1"))

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

@torch.no_grad()
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

@torch.no_grad()
def loss_like(model, x, t) -> float:
    # WARNING: touches BufferPool (forward)
    y = model(x)
    diff = (y.data - t.data)
    return float((diff * diff).mean().detach().cpu().item())


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=== PR1 verify: compile IR + lowering trace + determinism (full-state restore) ===")
    print(f"replay_n={REPLAY_N}, seed={SEED}, warmup_runs={WARMUP_RUNS}, print_loss_every={PRINT_LOSS_EVERY}")

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
    from aicf_fw.core.autograd import backward as autograd_backward
    from aicf_fw.core.warmup import warmup_capture_safe
    from aicf_fw.core.functional import functional_buffer_stats
    from aicf_fw.nn.linear import Linear
    from aicf_fw.nn.relu import ReLU
    from aicf_fw.nn.sequential import Sequential
    from aicf_fw.optim.adam import Adam
    from aicf_fw.core import functional as F
    from aicf_fw.core.compile import compile_ir, lower_to_backend_ops

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

    named_params = list(model.named_parameters())
    for n, p in named_params:
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    # Train step
    def train_step_aicf_only():
        optim.zero_grad()
        y = model(x)
        dY = F.mse_grad(y, t)
        autograd_backward(y, grad=dY, accumulate=False)
        optim.step_()

    # 1) Compile IR + lowering
    ir = compile_ir(train_step_aicf_only, name="train_step_aicf_only")
    if IR_DUMP:
        print("=== IR DUMP ===")
        print(ir.dump_json(indent=2))

    lowered = lower_to_backend_ops(ir)
    if LOWER_DUMP:
        print("=== LOWERED OPS ===")
        for i, it in enumerate(lowered):
            print(f"[lower {i:02d}] op={it['op']} attrs={it['attrs']}")

    # 2) Warmup (materialize buffers)
    warmup_capture_safe(train_step=train_step_aicf_only, runs=WARMUP_RUNS, sync=True)
    print("[warmup] functional buffers =", functional_buffer_stats())

    backend.capture_reset()
    torch.cuda.synchronize()

    # 3) Capture + trace
    backend.trace_reset()
    backend.trace_enable(True)

    backend.capture_begin()
    train_step_aicf_only()
    backend.capture_end()
    torch.cuda.synchronize()
    ok("[capture] done")

    trace_ops = backend.trace_get()  # List[str]
    backend.trace_enable(False)

    if TRACE_DUMP:
        print("=== TRACE OPS (runtime) ===")
        for i, op in enumerate(trace_ops):
            print(f"[trace {i:02d}] op={op}")

    # 4) Lowering vs runtime (v0 slices)
    lower_ops = [x["op"] for x in lowered]

    if TRACE_FILTER:
        IGNORE = {"grad_zero", "copy", "relu_bwd", "reduce_sum"}
        tr = [op for op in trace_ops if op not in IGNORE]
        lo = list(lower_ops)

        # forward slice until mse_grad
        tr_fw = tr[: tr.index("mse_grad") + 1] if "mse_grad" in tr else tr
        lo_fw = lo[: lo.index("mse_grad") + 1] if "mse_grad" in lo else lo
        if tr_fw != lo_fw:
            fail("Lowering mismatch (forward-slice):\n"
                 f"  trace_fw={tr_fw}\n  lower_fw={lo_fw}\n")

        # optim slice
        KEEP_OPT = {"step_inc", "bias_corr", "adam_step"}
        tr_opt = [op for op in tr if op in KEEP_OPT]
        lo_opt = [op for op in lo if op in KEEP_OPT]
        if tr_opt != lo_opt:
            fail("Lowering mismatch (optim-slice):\n"
                 f"  trace_opt={tr_opt}\n  lower_opt={lo_opt}\n")

        ok("[lowering] match: forward slice + optim slice")
    else:
        if trace_ops != lower_ops:
            fail("Lowering mismatch (strict):\n"
                 f"  trace_ops={trace_ops}\n  lower_ops={lower_ops}\n")
        ok("[lowering] strict match")

    # --------------------------------------------------------
    # 5) FULL TRAIN STATE snapshot/restore (CRITICAL)
    # --------------------------------------------------------
    @torch.no_grad()
    def snapshot_train_state():
        # params
        ps = {n: p.data.detach().clone() for n, p in model.named_parameters()}
        # grads (leaf grads are read by adam_step)
        gs = {}
        for n, p in model.named_parameters():
            gs[n] = None if p.grad is None else p.grad.data.detach().clone()
        # adam state
        ms = {i: optim.m[i].data.detach().clone() for i in optim.m.keys()}
        vs = {i: optim.v[i].data.detach().clone() for i in optim.v.keys()}
        step = optim.step.detach().clone()
        bc1 = optim.bc1_inv.detach().clone()
        bc2 = optim.bc2_inv.detach().clone()
        return ps, gs, ms, vs, step, bc1, bc2

    @torch.no_grad()
    def restore_train_state(st):
        ps, gs, ms, vs, step, bc1, bc2 = st
        cur = {n: p for n, p in model.named_parameters()}

        # params
        for n, src in ps.items():
            cur[n].data.copy_(src)

        # grads
        for n, g in gs.items():
            p = cur[n]
            if g is None:
                p.grad = None
            else:
                if p.grad is None:
                    # allocate once, OUTSIDE capture (we are outside capture here)
                    p.grad = Tensor(torch.empty_like(g), requires_grad=False)
                p.grad.data.copy_(g)

        # m/v
        for i in ms.keys():
            optim.m[i].data.copy_(ms[i])
            optim.v[i].data.copy_(vs[i])

        # scalars
        optim.step.copy_(step)
        optim.bc1_inv.copy_(bc1)
        optim.bc2_inv.copy_(bc2)

    @torch.no_grad()
    def assert_state_equal(st_ref, tag: str):
        ps, gs, ms, vs, step, bc1, bc2 = st_ref
        cur = {n: p for n, p in model.named_parameters()}

        for n, ref in ps.items():
            d = max_abs_diff(cur[n].data, ref)
            if d != 0.0:
                fail(f"[state] param mismatch {tag}: {n} maxdiff={d}")

        # grads equality (if both not None)
        for n, refg in gs.items():
            pg = cur[n].grad
            if refg is None:
                if pg is not None:
                    fail(f"[state] grad mismatch {tag}: {n} expected None")
            else:
                if pg is None:
                    fail(f"[state] grad mismatch {tag}: {n} expected tensor, got None")
                d = max_abs_diff(pg.data, refg)
                if d != 0.0:
                    fail(f"[state] grad mismatch {tag}: {n} maxdiff={d}")

        for i in ms.keys():
            dm = max_abs_diff(optim.m[i].data, ms[i])
            dv = max_abs_diff(optim.v[i].data, vs[i])
            if dm != 0.0:
                fail(f"[state] m mismatch {tag}: idx={i} maxdiff={dm}")
            if dv != 0.0:
                fail(f"[state] v mismatch {tag}: idx={i} maxdiff={dv}")

        if int(optim.step.item()) != int(step.item()):
            fail(f"[state] step mismatch {tag}: {int(optim.step.item())} != {int(step.item())}")
        if float(optim.bc1_inv.item()) != float(bc1.item()):
            fail(f"[state] bc1 mismatch {tag}: {optim.bc1_inv.item()} != {bc1.item()}")
        if float(optim.bc2_inv.item()) != float(bc2.item()):
            fail(f"[state] bc2 mismatch {tag}: {optim.bc2_inv.item()} != {bc2.item()}")

    # Snapshot AFTER capture (this is the real starting state for sequences)
    st0 = snapshot_train_state()

    # Run A
    stepdiff_A = []
    snaps = snapshot_params(model)
    for i in range(REPLAY_N):
        backend.replay()
        torch.cuda.synchronize()
        sd = max_param_diff(model, snaps)
        stepdiff_A.append(sd)
        snaps = snapshot_params(model)
        print(f"[A {i:02d}] stepdiff={sd:.6e}")
        if PRINT_LOSS_EVERY and (i % PRINT_LOSS_EVERY == 0):
            print(f"         loss_like={loss_like(model, x, t):.10f}")

    # Restore FULL STATE then run B
    restore_train_state(st0)
    torch.cuda.synchronize()
    if CHECK_RESTORE:
        assert_state_equal(st0, "after-restore")

    stepdiff_B = []
    snaps = snapshot_params(model)
    for i in range(REPLAY_N):
        backend.replay()
        torch.cuda.synchronize()
        sd = max_param_diff(model, snaps)
        stepdiff_B.append(sd)
        snaps = snapshot_params(model)
        print(f"[B {i:02d}] stepdiff={sd:.6e}")
        if PRINT_LOSS_EVERY and (i % PRINT_LOSS_EVERY == 0):
            print(f"         loss_like={loss_like(model, x, t):.10f}")

    # Compare
    bad = None
    for i, (a, b) in enumerate(zip(stepdiff_A, stepdiff_B)):
        if a != b:
            bad = i
            break

    if bad is not None:
        fail(f"Replay determinism broken (stepdiff sequence) at iter {bad:02d}: {stepdiff_A[bad]:.6e} != {stepdiff_B[bad]:.6e}")

    ok(f"Determinism OK: {REPLAY_N} replays stepdiff-sequence matches")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
