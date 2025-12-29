import os
import sys
from pathlib import Path
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
PYMOD_DIR = REPO_ROOT / "build" / "python"
PKG_DIR   = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))
if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import aicf_cuda as aicf


def op_call(kind, inputs, outputs, attrs=None):
    if attrs is None:
        attrs = {}
    aicf.op_call(kind, inputs, outputs, attrs)


def max_abs(a, b):
    return (a - b).abs().max().item()


def check_close(name, got, ref, atol=2e-5, rtol=2e-5):
    m = max_abs(got, ref)
    ok = torch.allclose(got, ref, atol=atol, rtol=rtol)
    print(f"[{name}] ok={ok} max_abs={m} atol={atol} rtol={rtol}")
    if not ok:
        raise RuntimeError(f"{name} mismatch")


@torch.inference_mode()
def test_adam_biascorr_matches_torch_auto(steps=20, shape=(1024,)):
    """
    Your AdamStep expects attrs: bc1_inv, bc2_inv.
    But BiasCorr output semantics might be:
      - (1 - beta^t) or its inverse, or sqrt variant, etc.

    We auto-pick the interpretation that best matches torch Adam at t=1.
    """
    assert torch.cuda.is_available()
    torch.manual_seed(0)

    lr    = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps   = 1e-8

    # AICF tensors
    p  = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    g0 = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()  # fixed grad
    m  = torch.zeros_like(p)
    v  = torch.zeros_like(p)

    step = torch.zeros((), device="cuda", dtype=torch.int32)
    bc1  = torch.empty((), device="cuda", dtype=torch.float32)
    bc2  = torch.empty((), device="cuda", dtype=torch.float32)

    # torch reference
    p_ref = p.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([p_ref], lr=lr, betas=(beta1, beta2), eps=eps)

    # --- helper: run exactly one AICF step given bc1_inv/bc2_inv ---
    def aicf_step_with(bc1_inv, bc2_inv):
        attrs = {
            "lr": float(lr),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "eps": float(eps),
            "bc1_inv": float(bc1_inv),
            "bc2_inv": float(bc2_inv),
        }
        op_call(aicf.OpKind.AdamStep, [p, g0, m, v], [p, m, v], attrs)

    # ---- Step 1: find best interpretation at t=1 ----
    # torch t=1
    p_ref.grad = g0.detach().clone()
    opt.step()
    p_t1 = p_ref.detach().clone()

    # AICF: produce bc outputs for t=1
    op_call(aicf.OpKind.StepInc,  [step],     [step],     {})
    op_call(aicf.OpKind.BiasCorr, [step],     [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})

    b1 = float(bc1.item())
    b2 = float(bc2.item())

    # Candidates: map (b1,b2) -> (bc1_inv, bc2_inv)
    # We try common possibilities:
    # 0) already inv
    # 1) inv of value
    # 2) sqrt-inv of value
    # 3) value itself but clamp to avoid zero
    cand = []
    eps_safe = 1e-20

    cand.append(("as_is",            b1,                      b2))
    cand.append(("inv",              1.0 / max(b1, eps_safe), 1.0 / max(b2, eps_safe)))
    cand.append(("sqrt_inv",         1.0 / (max(b1, eps_safe) ** 0.5),
                                 1.0 / (max(b2, eps_safe) ** 0.5)))
    cand.append(("inv_sqrt",         (max(b1, eps_safe) ** 0.5) and (1.0 / (max(b1, eps_safe) ** 0.5)),
                                 (max(b2, eps_safe) ** 0.5) and (1.0 / (max(b2, eps_safe) ** 0.5))))
    # duplicate-safe (inv_sqrt == sqrt_inv), keep anyway harmless

    # Also try "torch-style denom" guess:
    # if BiasCorr returns (1 - beta^t), then inv should be 1/(1-beta^t)
    # if it returns 1/(1-beta^t), then as_is should work.
    # already covered by as_is/inv.

    # Evaluate each candidate by cloning state and running one AICF step
    best = None
    best_err = None

    # save initial state
    p0 = p.detach().clone()
    m0 = m.detach().clone()
    v0 = v.detach().clone()

    for name, bc1_inv, bc2_inv in cand:
        # restore
        p.copy_(p0)
        m.copy_(m0)
        v.copy_(v0)

        aicf_step_with(bc1_inv, bc2_inv)
        torch.cuda.synchronize()

        err = max_abs(p, p_t1)
        print(f"[probe t=1] cand={name:8s} bc1_inv={bc1_inv:.6e} bc2_inv={bc2_inv:.6e} max_abs={err:.6e}")
        if best_err is None or err < best_err:
            best_err = err
            best = (name, bc1_inv, bc2_inv)

    if best is None:
        raise RuntimeError("No biascorr interpretation candidate worked.")

    sel_name, sel_bc1_inv, sel_bc2_inv = best
    print(f"[selected] {sel_name} (t=1 err={best_err:.6e})")

    # Now run full steps with selected interpretation.
    # Reset everything to initial.
    p.copy_(p0); m.copy_(m0); v.copy_(v0)
    p_ref = p0.detach().clone().requires_grad_(True)
    opt = torch.optim.Adam([p_ref], lr=lr, betas=(beta1, beta2), eps=eps)

    # reset step on device
    step.zero_()

    for t in range(steps):
        # torch
        p_ref.grad = g0.detach().clone()
        opt.step()

        # aicf biascorr update each step (we still *use* selected mapping)
        op_call(aicf.OpKind.StepInc,  [step],     [step],     {})
        op_call(aicf.OpKind.BiasCorr, [step],     [bc1, bc2], {"beta1": float(beta1), "beta2": float(beta2)})

        b1 = float(bc1.item())
        b2 = float(bc2.item())

        # apply chosen mapping
        if sel_name == "as_is":
            bc1_inv, bc2_inv = b1, b2
        elif sel_name == "inv":
            bc1_inv, bc2_inv = 1.0 / max(b1, eps_safe), 1.0 / max(b2, eps_safe)
        elif sel_name in ("sqrt_inv", "inv_sqrt"):
            bc1_inv, bc2_inv = 1.0 / (max(b1, eps_safe) ** 0.5), 1.0 / (max(b2, eps_safe) ** 0.5)
        else:
            bc1_inv, bc2_inv = sel_bc1_inv, sel_bc2_inv  # fallback constant

        attrs = {
            "lr": float(lr),
            "beta1": float(beta1),
            "beta2": float(beta2),
            "eps": float(eps),
            "bc1_inv": float(bc1_inv),
            "bc2_inv": float(bc2_inv),
        }
        op_call(aicf.OpKind.AdamStep, [p, g0, m, v], [p, m, v], attrs)

    torch.cuda.synchronize()
    check_close("AdamStep+BiasCorr vs torch Adam (auto)", p, p_ref.detach(), atol=2e-5, rtol=2e-5)


@torch.inference_mode()
def test_adam_biascorr_aicf_graph_replay_no_crash(replays=200, shape=(4096,)):
    """
    Capture-safe crash-only test (biascorr changing attrs is not capture-safe as-is).
    """
    lr    = 1e-3
    beta1 = 0.9
    beta2 = 0.999
    eps   = 1e-8

    p  = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    g  = torch.randn(*shape, device="cuda", dtype=torch.float32).contiguous()
    m  = torch.zeros_like(p)
    v  = torch.zeros_like(p)

    # fixed correction for capture
    t = 1
    bc1_inv = 1.0 / (1.0 - (beta1 ** t))
    bc2_inv = 1.0 / (1.0 - (beta2 ** t))

    attrs = {
        "lr": float(lr),
        "beta1": float(beta1),
        "beta2": float(beta2),
        "eps": float(eps),
        "bc1_inv": float(bc1_inv),
        "bc2_inv": float(bc2_inv),
    }

    for _ in range(5):
        op_call(aicf.OpKind.AdamStep, [p, g, m, v], [p, m, v], attrs)
    torch.cuda.synchronize()

    aicf.capture_reset()
    torch.cuda.synchronize()

    aicf.capture_begin()
    op_call(aicf.OpKind.AdamStep, [p, g, m, v], [p, m, v], attrs)
    aicf.capture_end()
    torch.cuda.synchronize()

    for _ in range(replays):
        aicf.replay()
    torch.cuda.synchronize()

    print("[AdamStep(AICF Graph)] replay OK (no crash)")


if __name__ == "__main__":
    test_adam_biascorr_matches_torch_auto(steps=20, shape=(1024,))
    test_adam_biascorr_aicf_graph_replay_no_crash(replays=200, shape=(4096,))
    print("ALL OK")
