from __future__ import annotations

import sys
from pathlib import Path
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

    # ✅ fw-style API: model.compile(...)
    model.compile(
        optimizer=opt,
        B=B, D=D, device=device, dtype=dtype,
        name="fw_mvp_train_step",
        warmup_runs=2,
        warmup_inputs={"x": x, "t": t},
        warmup_required=True,
    )

    W0 = dict(model.named_parameters())["0.W"]
    W0_before = W0.clone()

    # --- train_step (compiled handle train_step) ---
    model.train_step({"x": x, "t": t})
    dW0_1 = maxabs_delta(W0, W0_before)
    print("[train_step] |ΔW0| =", dW0_1)
    if dW0_1 == 0.0:
        raise RuntimeError("W0 did not update on train_step")

    # --- capture + replay ---
    model.capture({"x": x, "t": t})
    print("OK (capture)")

    W0_cap0 = W0.clone()
    model.replay(n=3)
    dW0_rep = maxabs_delta(W0, W0_cap0)
    print("[replay] n=3 |ΔW0| =", dW0_rep)
    if dW0_rep == 0.0:
        raise RuntimeError("W0 did not update across replay(n=3)")

    # --- meta mutation sanity ---
    bc1 = opt.bc1_inv
    bc2 = opt.bc2_inv
    bc1_before = float(bc1.item())
    bc2_before = float(bc2.item())

    W0_m0 = W0.clone()
    bc1.fill_(1.0)
    bc2.fill_(1.0)
    torch.cuda.synchronize()
    model.replay(n=1)
    d_mut = maxabs_delta(W0, W0_m0)

    W0_m1 = W0.clone()
    bc1.fill_(bc1_before)
    bc2.fill_(bc2_before)
    torch.cuda.synchronize()
    model.replay(n=1)
    d_rest = maxabs_delta(W0, W0_m1)

    print("[meta] mutated |ΔW0| =", d_mut, " restored |ΔW0| =", d_rest)
    if abs(d_mut - d_rest) < 1e-12:
        raise RuntimeError("meta mutation did not change replay behavior")

    model.reset()
    print("ALL OK (fw MVP user experience test)")


if __name__ == "__main__":
    main()
