# examples/python/python_framework_test/verify_pr3.py
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
DTYPE_STR = os.environ.get("AICF_DTYPE", "float32").lower()
ATOL = float(os.environ.get("AICF_ATOL", "1e-5"))
RTOL = float(os.environ.get("AICF_RTOL", "1e-5"))


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


def torch_dtype():
    if DTYPE_STR in ("fp32", "float32"):
        return torch.float32
    if DTYPE_STR in ("fp16", "float16", "half"):
        return torch.float16
    fail(f"Unsupported dtype: {DTYPE_STR}")


@torch.no_grad()
def max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


@torch.no_grad()
def assert_allclose(name: str, a: torch.Tensor, b: torch.Tensor, atol=ATOL, rtol=RTOL):
    if a.shape != b.shape:
        fail(f"{name}: shape mismatch {tuple(a.shape)} vs {tuple(b.shape)}")
    a_ = a.float() if a.dtype != b.dtype else a
    b_ = b.float() if a.dtype != b.dtype else b
    if not torch.allclose(a_, b_, atol=atol, rtol=rtol):
        md = max_abs_diff(a_.float(), b_.float())
        fail(f"{name}: allclose FAIL max|diff|={md:.3e} (atol={atol:g}, rtol={rtol:g})")
    md = max_abs_diff(a_.float(), b_.float())
    print(f"[OK] {name}: allclose PASS max|diff|={md:.3e}")


def torch_mse_loss(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return ((y - t) ** 2).mean()


def torch_mse_dy(y: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    numel = y.numel()
    return 2.0 * (y - t) / float(numel)


# ------------------------------------------------------------
# Backend setup
# ------------------------------------------------------------
def setup_aicf_backend():
    from aicf_fw.backend.aicf_backend import AICFBackend
    from aicf_fw.backend import set_backend

    backend = AICFBackend()
    set_backend(backend)
    ok(f"Backend set: {type(backend).__name__}")
    return backend


# ------------------------------------------------------------
# Op sanity tests
# ------------------------------------------------------------
def test_op_gemm_transpose_sanity():
    from aicf_fw.backend import set_backend, get_backend
    from aicf_fw.backend.aicf_backend import AICFBackend

    # Ensure backend is set (idempotent)
    try:
        bk = get_backend()
    except AssertionError:
        backend = AICFBackend()
        set_backend(backend)
        bk = get_backend()

    device = "cuda"
    dtype = torch.float32

    # A: (B, IN), B: (B, OUT)  -> want: A^T @ B = (IN, OUT)
    Bsz, IN, OUT = 7, 5, 3
    A = torch.randn(Bsz, IN, device=device, dtype=dtype)
    Bm = torch.randn(Bsz, OUT, device=device, dtype=dtype)

    out = torch.empty(IN, OUT, device=device, dtype=dtype)
    bk.op_call_out("gemm", [A, Bm], [out], {"transA": True, "transB": False})

    ref = A.t() @ Bm
    assert_allclose("op:gemm transA sanity", out, ref)


def test_op_copy_sanity():
    from aicf_fw.backend import get_backend
    bk = get_backend()

    x = torch.randn(257, device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)
    bk.op_call_out("copy", [x], [y], {})
    assert_allclose("op:copy sanity", y, x)


def test_op_add_inplace_sanity():
    from aicf_fw.backend import get_backend
    bk = get_backend()

    a = torch.randn(333, device="cuda", dtype=torch.float32)
    b = torch.randn_like(a)
    ref = a + b
    bk.op_call_out("add", [a, b], [a], {})
    assert_allclose("op:add in-place sanity", a, ref)


def test_op_relu_bwd_sanity():
    """
    relu_bwd(dout, y) should match: dout * (x > 0)
    Here we compute y = relu(x) and compare dx.
    """
    from aicf_fw.backend import get_backend
    bk = get_backend()

    x = torch.randn(1024, device="cuda", dtype=torch.float32)
    y = torch.relu(x)
    dout = torch.randn_like(x)

    dx = torch.empty_like(x)
    bk.op_call_out("relu_bwd", [dout, y], [dx], {})

    ref = dout * (x > 0).to(dout.dtype)
    assert_allclose("op:relu_bwd sanity", dx, ref)


# ------------------------------------------------------------
# Framework-level tests
# ------------------------------------------------------------
def test_op_mse_grad():
    from aicf_fw.core.tensor import Tensor
    from aicf_fw.core import functional as F

    dtype = torch_dtype()
    device = "cuda"

    y = torch.randn(64, 8, device=device, dtype=dtype)
    t = torch.randn(64, 8, device=device, dtype=dtype)

    y_a = Tensor(y, requires_grad=False)
    t_a = Tensor(t, requires_grad=False)

    dY_a = F.mse_grad(y_a, t_a)
    if not hasattr(dY_a, "data") or not torch.is_tensor(dY_a.data):
        fail("F.mse_grad must return a Tensor with .data being torch.Tensor")

    dY_t = torch_mse_dy(y, t)
    assert_allclose("op:mse_grad (dL/dy)", dY_a.data, dY_t)
    ok("op:mse_grad matches torch golden")


def test_one_step_train_correctness():
    from aicf_fw.core.tensor import Tensor
    from aicf_fw.core.autograd import backward as autograd_backward
    from aicf_fw.nn.linear import Linear
    from aicf_fw.nn.relu import ReLU
    from aicf_fw.nn.sequential import Sequential
    from aicf_fw.optim.adam import Adam
    from aicf_fw.core import functional as F

    dtype = torch_dtype()
    device = "cuda"

    # Torch model (expanded forward for dout0 extraction)
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(8, 8, bias=True, device=device, dtype=dtype),
        torch.nn.ReLU(),
        torch.nn.Linear(8, 8, bias=True, device=device, dtype=dtype),
    )

    # AICF model
    aicf_model = Sequential(
        Linear(8, 8, device=device, dtype=dtype),
        ReLU(),
        Linear(8, 8, device=device, dtype=dtype),
    )

    # Copy torch params -> aicf params
    with torch.no_grad():
        for n, p in aicf_model.named_parameters():
            if n == "0.W":
                p.data.copy_(torch_model[0].weight)
            elif n == "0.b":
                p.data.copy_(torch_model[0].bias)
            elif n == "2.W":
                p.data.copy_(torch_model[2].weight)
            elif n == "2.b":
                p.data.copy_(torch_model[2].bias)
            else:
                warn(f"Unknown AICF param name: {n}")

    # Input/target
    x = torch.randn(64, 8, device=device, dtype=dtype)
    t = torch.randn(64, 8, device=device, dtype=dtype)

    # ---- Torch: get dout0 and grads ----
    torch_opt = torch.optim.Adam(torch_model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    torch_opt.zero_grad(set_to_none=True)

    x_t = x.detach().clone().requires_grad_(True)
    h0 = torch_model[0](x_t)
    h0.retain_grad()
    r0 = torch.relu(h0)
    y_t = torch_model[2](r0)
    loss_t = torch_mse_loss(y_t, t)
    loss_t.backward()

    dout0 = h0.grad.detach().clone()  # (B, OUT)
    torch_opt.step()

    with torch.no_grad():
        t_0w = torch_model[0].weight.detach().clone()
        t_0b = torch_model[0].bias.detach().clone()
        t_2w = torch_model[2].weight.detach().clone()
        t_2b = torch_model[2].bias.detach().clone()

        g_0w = torch_model[0].weight.grad.detach().clone()
        g_0b = torch_model[0].bias.grad.detach().clone()
        g_2w = torch_model[2].weight.grad.detach().clone()
        g_2b = torch_model[2].bias.grad.detach().clone()

    # ---- AICF: forward/backward/step ----
    aicf_opt = Adam(aicf_model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8, grad_clip=None)

    x_a = Tensor(x, requires_grad=False)
    t_a = Tensor(t, requires_grad=False)

    aicf_opt.zero_grad()
    y_a = aicf_model(x_a)

    loss_a = ((y_a.data - t) ** 2).mean()
    dY = F.mse_grad(y_a, t_a)
    autograd_backward(y_a, grad=dY, accumulate=False)
    aicf_opt.step_()

    # Snapshot AICF params + grads
    with torch.no_grad():
        a_params = {}
        a_grads = {}
        for n, p in aicf_model.named_parameters():
            a_params[n] = p.data.detach().clone()
            if p.grad is None:
                a_grads[n] = None
            else:
                gd = p.grad.data if hasattr(p.grad, "data") else p.grad
                a_grads[n] = gd.detach().clone() if torch.is_tensor(gd) else None

    # Forward y and loss
    assert_allclose("train1:forward y", y_a.data, y_t.detach())
    la = float(loss_a.detach().cpu().item())
    lt = float(loss_t.detach().cpu().item())
    if abs(la - lt) > 1e-6:
        fail(f"train1:loss scalar mismatch {la:.10f} != {lt:.10f}")
    ok(f"train1:loss scalar match ({la:.10f})")

    # Grads
    if any(a_grads[k] is None for k in ("0.W", "0.b", "2.W", "2.b")):
        warn("Some AICF grads are None")
    else:
        # DIAG: compare aicf 0.W grad to torch-derived formula
        dW0_ref = dout0.t() @ x  # (OUT, IN)
        a0 = a_grads["0.W"]

        print(f"[diag] dW0_ref max|.|={float(dW0_ref.abs().max().item()):.3e}")
        print(f"[diag] a0       max|.|={float(a0.abs().max().item()):.3e}")
        print(f"[diag] a0 vs dW0_ref    max|diff|={float((a0 - dW0_ref).abs().max().item()):.3e}")
        print(f"[diag] a0 vs dW0_ref.T  max|diff|={float((a0 - dW0_ref.t()).abs().max().item()):.3e}")

        assert_allclose("train1:grad 0.W", a0, g_0w)
        assert_allclose("train1:grad 0.b", a_grads["0.b"], g_0b)
        assert_allclose("train1:grad 2.W", a_grads["2.W"], g_2w)
        assert_allclose("train1:grad 2.b", a_grads["2.b"], g_2b)

    # Params after 1 Adam step
    assert_allclose("train1:param 0.W", a_params["0.W"], t_0w)
    assert_allclose("train1:param 0.b", a_params["0.b"], t_0b)
    assert_allclose("train1:param 2.W", a_params["2.W"], t_2w)
    assert_allclose("train1:param 2.b", a_params["2.b"], t_2b)

    ok("one-step train correctness matches torch golden")


def main():
    # Basic CUDA availability
    if not torch.cuda.is_available():
        fail("CUDA not available")

    # Backend + seed
    seed_all(SEED)
    setup_aicf_backend()

    # Op sanity first (pinpoint culprit fast)
    test_op_gemm_transpose_sanity()
    test_op_copy_sanity()
    test_op_add_inplace_sanity()
    test_op_relu_bwd_sanity()

    print("=== PR3 verify: torch golden numerical correctness ===")
    print(f"seed={SEED} dtype={DTYPE_STR} atol={ATOL:g} rtol={RTOL:g}")

    # Op-level test
    seed_all(SEED)
    test_op_mse_grad()

    # One-step golden test (re-seed to align init)
    seed_all(SEED)
    test_one_step_train_correctness()

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
