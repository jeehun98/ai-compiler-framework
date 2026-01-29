from __future__ import annotations

import sys
from pathlib import Path
import time
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.nn import Sequential, Linear, ReLU
from aicf_fw.optim import Adam


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def require_nonzero(x: float, msg: str):
    if x == 0.0:
        raise RuntimeError(msg)


def main():
    tf32_off()
    torch.manual_seed(0)

    device = "cuda:0"
    dtype = torch.float32
    B, D = 64, 8

    x = torch.randn(B, D, device=device, dtype=dtype)
    t = torch.randn(B, D, device=device, dtype=dtype)

    model = Sequential(
        Linear(D, D, bias=True, device=device, dtype=dtype),
        ReLU(),
        Linear(D, D, bias=True, device=device, dtype=dtype),
    ).to(device)

    opt = Adam(model, lr=1e-3)

    art_root = ensure_dir(Path("artifacts") / f"{now_tag()}_fw_emit_adam_train_step_test")

    # ---- compile ----
    model.compile(
        optimizer=opt,
        B=B, D=D, device=device, dtype=dtype,
        name="fw_emit_adam_train_step_test",
        warmup_runs=2,
        warmup_inputs={"x": x, "t": t},
        warmup_required=True,
    )

    if not model.is_compiled():
        raise RuntimeError("model.compile() did not attach compiled handle")

    compiled = model._compiled  # 내부 핸들 접근(테스트에서만)
    # compiled: CompiledTrainStep

    # ---- dump lowered/plan if available ----
    # (core_v2 dump util이 있으면 그걸 쓰고, 없으면 그냥 repr로)
    try:
        from aicf_fw.core_v2 import dump_lowered, dump_plan
        lowered_txt = dump_lowered(compiled.lowered, name="fw_emit_adam_train_step_test")
        dump_text(art_root / "30_lowered.txt", lowered_txt)

        plan_txt = dump_plan(compiled.plan, name="fw_emit_adam_train_step_test")
        dump_text(art_root / "40_plan.txt", plan_txt)
    except Exception as e:
        dump_text(art_root / "30_lowered_fallback.txt", repr(compiled.lowered))
        dump_text(art_root / "40_plan_fallback.txt", repr(compiled.plan))
        dump_text(art_root / "00_dump_warn.txt", f"dump_lowered/dump_plan not available: {e}")

    # ---- train_step ----
    W0 = dict(model.named_parameters())["0.W"]
    W0_before = W0.clone()

    model.train_step({"x": x, "t": t})
    dW0_1 = maxabs_delta(W0, W0_before)
    print("[train_step] |ΔW0| =", dW0_1)
    require_nonzero(dW0_1, "W0 did not update on train_step")

    # ---- capture + replay ----
    model.capture({"x": x, "t": t})
    print("OK (capture)")

    W0_cap0 = W0.clone()
    model.replay(n=3, sync=True)
    dW0_rep = maxabs_delta(W0, W0_cap0)
    print("[replay] n=3 |ΔW0| =", dW0_rep)
    require_nonzero(dW0_rep, "W0 did not update across replay(n=3)")

    # ---- meta mutation sanity ----
    bc1 = opt.bc1_inv
    bc2 = opt.bc2_inv
    bc1_before = float(bc1.item())
    bc2_before = float(bc2.item())

    W0_m0 = W0.clone()
    bc1.fill_(1.0)
    bc2.fill_(1.0)
    model.replay(n=1, sync=True)
    d_mut = maxabs_delta(W0, W0_m0)

    W0_m1 = W0.clone()
    bc1.fill_(bc1_before)
    bc2.fill_(bc2_before)
    model.replay(n=1, sync=True)
    d_rest = maxabs_delta(W0, W0_m1)

    print("[meta] mutated |ΔW0| =", d_mut, " restored |ΔW0| =", d_rest)
    if abs(d_mut - d_rest) < 1e-12:
        raise RuntimeError("meta mutation did not change replay behavior")

    # ---- async vs sync ----
    W0_a0 = W0.clone()
    model.replay(n=1, sync=False)
    d_async = maxabs_delta(W0, W0_a0)

    W0_s0 = W0.clone()
    model.replay(n=1, sync=True)
    d_sync = maxabs_delta(W0, W0_s0)

    print("[replay sync-check] async |ΔW0| =", d_async, " sync |ΔW0| =", d_sync)
    require_nonzero(d_sync, "sync replay did not update W0")

    # ---- runtime trace dump (if executor exposes it) ----
    try:
        ex = compiled.ex
        if hasattr(ex, "trace_reset") and hasattr(ex, "trace_enable") and hasattr(ex, "trace_get"):
            ex.trace_reset()
            ex.trace_enable(True)
            model.train_step({"x": x, "t": t})
            ex.trace_enable(False)
            trace = "\n".join([str(s) for s in ex.trace_get()])
            print("=== TRACE ===")
            print(trace)
            dump_text(art_root / "50_runtime_trace.txt", trace)
        else:
            dump_text(art_root / "50_runtime_trace.txt", "executor has no trace_* api")
    except Exception as e:
        dump_text(art_root / "50_runtime_trace_error.txt", repr(e))

    model.reset()
    print(f"[OK] artifacts dumped to: {art_root}")
    print("ALL OK (fw emit+adam train_step test)")


if __name__ == "__main__":
    main()
