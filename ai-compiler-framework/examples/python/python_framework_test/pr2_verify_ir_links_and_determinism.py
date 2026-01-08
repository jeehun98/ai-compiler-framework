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

PRINT_LOSS_EVERY = int(os.environ.get("AICF_PRINT_LOSS_EVERY", "0"))  # 0=never
CHECK_RESTORE = int(os.environ.get("AICF_CHECK_RESTORE", "1"))

# IR structural checks
CHECK_IR_LINKS = int(os.environ.get("AICF_CHECK_IR_LINKS", "1"))
CHECK_IR_NO_SCALAR_BLOWUP = int(os.environ.get("AICF_CHECK_IR_NO_SCALAR_BLOWUP", "1"))


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
# IR checks
# ------------------------------------------------------------
def _ir_values_by_name(ir) -> dict[str, list[int]]:
    # ir.values: Dict[int, IRValue]
    out: dict[str, list[int]] = {}
    for vid, v in ir.values.items():
        out.setdefault(v.name, []).append(int(vid))
    return out

def _ir_node(ir, op: str):
    for n in ir.nodes:
        if n.op == op:
            return n
    return None

def check_ir_links(ir):
    """
    1) Backward input should be SAME IRValue id as the tensor produced by MseGrad (dY),
       not a fresh unrelated value.
    2) Backward input "loss" should point to the model output value id (Linear2 output),
       not a random new value.
    """
    bw = _ir_node(ir, "Backward")
    if bw is None:
        fail("IR has no Backward node")

    if len(bw.inputs) < 1:
        fail("Backward node has no inputs")

    # Find MseGrad node output id
    mg = _ir_node(ir, "MseGrad")
    if mg is None:
        fail("IR has no MseGrad node")
    if len(mg.outputs) != 1:
        fail(f"MseGrad outputs expected 1, got {len(mg.outputs)}")
    dy_id = int(mg.outputs[0])

    # Backward second input should be dy_id (if grad provided)
    if len(bw.inputs) >= 2:
        bw_dy = int(bw.inputs[1])
        if bw_dy != dy_id:
            fail(
                "IR link mismatch: Backward(grad) must reference MseGrad output.\n"
                f"  backward.grad={bw_dy}, mse_grad.out={dy_id}"
            )
    else:
        warn("Backward has no explicit grad input (your train_step probably passed grad=None?)")

    # Backward first input should reference the same output id as final forward value used for mse_grad input.
    # In your IR, MseGrad inputs: [pred, target]. pred is output of second Linear.
    pred_id = int(mg.inputs[0])
    bw_loss = int(bw.inputs[0])
    if bw_loss != pred_id:
        fail(
            "IR link mismatch: Backward(loss) must reference MseGrad pred input (model output).\n"
            f"  backward.loss={bw_loss}, mse_grad.pred={pred_id}"
        )

    ok("[ir] links OK: Backward(loss,grad) are connected to forward graph")

def check_ir_no_scalar_blowup(ir):
    """
    Optim scalars step/bc1_inv/bc2_inv should not explode into many IRValues.
    With proper caching, each should be a small constant count (typically 1 each).
    """
    by_name = _ir_values_by_name(ir)
    step_n = len(by_name.get("step", []))
    b1_n = len(by_name.get("bc1_inv", []))
    b2_n = len(by_name.get("bc2_inv", []))

    # tolerate a small number (some code paths may materialize 2-3),
    # but if it's ~#AdamSteps, it's broken.
    if step_n > 4 or b1_n > 4 or b2_n > 4:
        fail(
            "IR scalar blowup detected (trace cache likely broken).\n"
            f"  step={step_n}, bc1_inv={b1_n}, bc2_inv={b2_n} (expected <=4 each)"
        )
    ok(f"[ir] scalar caching OK: step={step_n}, bc1_inv={b1_n}, bc2_inv={b2_n}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    print("=== PR2 verify: IR links + lowering + runtime trace + determinism(full restore) ===")
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

    for n, p in model.named_parameters():
        print("[param]", n, tuple(p.data.shape), p.data.dtype, p.data.device)

    # Train step
    def train_step_aicf_only():
        optim.zero_grad()
        y = model(x)
        dY = F.mse_grad(y, t)
        autograd_backward(y, grad=dY, accumulate=False)
        optim.step_()

    # 1) Compile IR + IR checks
    ir = compile_ir(train_step_aicf_only, name="train_step_aicf_only")
    if IR_DUMP:
        print("=== IR DUMP ===")
        print(ir.dump_json(indent=2))

    if CHECK_IR_LINKS:
        check_ir_links(ir)
    if CHECK_IR_NO_SCALAR_BLOWUP:
        check_ir_no_scalar_blowup(ir)

    # 2) Lowering
    lowered = lower_to_backend_ops(ir)

    # Guard: catch accidental tuple return `return lowered,`
    if isinstance(lowered, tuple):
        fail(
            "lower_to_backend_ops() returned tuple. "
            "You probably have `return lowered,` bug. "
            f"type={type(lowered)} len={len(lowered)}"
        )
    if not isinstance(lowered, list):
        fail(f"lower_to_backend_ops() must return list[dict], got {type(lowered)}")

    if LOWER_DUMP:
        print("=== LOWERED OPS ===")
        for i, it in enumerate(lowered):
            print(f"[lower {i:02d}] op={it['op']} attrs={it['attrs']}")

    # 3) Warmup (materialize buffers + leaf grads)
    warmup_capture_safe(train_step=train_step_aicf_only, runs=WARMUP_RUNS, sync=True)
    print("[warmup] functional buffers =", functional_buffer_stats())

    backend.capture_reset()
    torch.cuda.synchronize()

    # 4) Capture + runtime trace
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

    # 5) Lowering vs runtime trace
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
    # 6) FULL TRAIN STATE snapshot/restore (CRITICAL)
    # --------------------------------------------------------
    @torch.no_grad()
    def snapshot_train_state():
        ps = {n: p.data.detach().clone() for n, p in model.named_parameters()}
        gs = {}
        for n, p in model.named_parameters():
            gs[n] = None if p.grad is None else p.grad.data.detach().clone()
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

        for n, src in ps.items():
            cur[n].data.copy_(src)

        for n, g in gs.items():
            p = cur[n]
            if g is None:
                p.grad = None
            else:
                if p.grad is None:
                    # OUTSIDE capture only
                    p.grad = Tensor(torch.empty_like(g), requires_grad=False)
                p.grad.data.copy_(g)

        for i in ms.keys():
            optim.m[i].data.copy_(ms[i])
            optim.v[i].data.copy_(vs[i])

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
    for i, (a, b) in enumerate(zip(stepdiff_A, stepdiff_B)):
        if a != b:
            fail(
                "Replay determinism broken (stepdiff sequence).\n"
                f"  iter={i:02d} A={a:.6e} B={b:.6e}\n"
                f"  A_seq={stepdiff_A}\n"
                f"  B_seq={stepdiff_B}"
            )

    ok(f"Determinism OK: {REPLAY_N} replays stepdiff-sequence matches")
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
