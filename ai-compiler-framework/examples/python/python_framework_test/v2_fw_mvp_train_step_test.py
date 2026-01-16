from __future__ import annotations

import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.nn import Sequential, Linear, ReLU
from aicf_fw.optim import Adam
from aicf_fw.fw import compile_train_step


def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def main():
    tf32_off()
    torch.manual_seed(0)

    device = "cuda:0"
    dtype = torch.float32
    B, D = 64, 8

    # inputs (runtime)
    x = torch.randn(B, D, device=device, dtype=dtype)
    t = torch.randn(B, D, device=device, dtype=dtype)

    # --- user-facing model build ---
    model = Sequential(
        Linear(D, D, bias=True, device=device, dtype=dtype),
        ReLU(),
        Linear(D, D, bias=True, device=device, dtype=dtype),
    ).to(device)

    # --- user-facing optimizer ---
    opt = Adam(model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8)

    # --- compile into core_v2 executor ---
    compiled = compile_train_step(model, opt, B=B, D=D, device=device, dtype=dtype, name="fw_mvp_train_step")

    # ============
    # 1) eager step
    # ============
    pnames = [name for name, _ in model.named_parameters()]
    assert "0.W" in pnames and "0.b" in pnames and "2.W" in pnames and "2.b" in pnames, f"param names unexpected: {pnames}"

    W0 = dict(model.named_parameters())["0.W"]
    W0_before = W0.clone()

    compiled.train_step({"x": x, "t": t})
    dW0_1 = maxabs_delta(W0, W0_before)
    print("[eager] |ΔW0| =", dW0_1)
    if dW0_1 == 0.0:
        raise RuntimeError("W0 did not update on eager train_step (expected non-zero)")

    print("OK (eager train_step updates params)")

    # =================
    # 2) capture/replay
    # =================
    compiled.capture({"x": x, "t": t})
    print("OK (capture)")

    W0_cap0 = W0.clone()
    compiled.replay(n=3)
    dW0_rep = maxabs_delta(W0, W0_cap0)
    print("[replay] n=3 |ΔW0| =", dW0_rep)
    if dW0_rep == 0.0:
        raise RuntimeError("W0 did not update across replay(n=3) (expected non-zero)")

    print("OK (replay updates params)")

    # ==========================================
    # 3) meta mutation sanity (optional but useful)
    # ==========================================
    # Change optimizer meta scalars and confirm it affects update magnitude in replay.
    # (This assumes your executor reads bc1/bc2 from the bound tensors each replay.)
    bc1 = opt.bc1_inv
    bc2 = opt.bc2_inv

    # snapshot
    bc1_before = float(bc1.item())
    bc2_before = float(bc2.item())

    W0_m0 = W0.clone()
    bc1.fill_(1.0)
    bc2.fill_(1.0)
    torch.cuda.synchronize()
    compiled.replay(n=1)
    d_mut = maxabs_delta(W0, W0_m0)

    W0_m1 = W0.clone()
    bc1.fill_(bc1_before)
    bc2.fill_(bc2_before)
    torch.cuda.synchronize()
    compiled.replay(n=1)
    d_rest = maxabs_delta(W0, W0_m1)

    print("[meta] mutated |ΔW0| =", d_mut, " restored |ΔW0| =", d_rest)
    if abs(d_mut - d_rest) < 1e-12:
        raise RuntimeError("meta mutation did not change replay behavior (unexpected)")

    print("OK (meta affects replay updates)")

    # reset graph (optional)
    compiled.reset()
    print("OK (reset)")

    print("ALL OK (fw MVP user experience test)")


if __name__ == "__main__":
    main()
