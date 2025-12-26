from __future__ import annotations

import os
import sys
from pathlib import Path

# --- AICF build/python bootstrap (for aicf_cuda) ---
REPO_ROOT = Path(__file__).resolve().parents[2]
PYMOD_DIR = REPO_ROOT / "build" / "python"
PKG_DIR = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))

if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))
# -----------------------------------------------

import torch

from aicf_fw.tensor import Tensor
from aicf_fw.modules.sequential import Sequential
from aicf_fw.modules.linear import Linear
from aicf_fw.modules.activations import ReLU
from aicf_fw.losses.mse import MSE
from aicf_fw.optim.sgd import SGD
from aicf_fw.trainer import Trainer, TrainerConfig
import aicf_fw.backend as B


def _env(name: str, default: str) -> str:
    v = os.environ.get(name, default)
    return default if v is None or v == "" else v


def _torch_dtype(name: str) -> torch.dtype:
    n = name.lower().strip()
    if n in ("f32", "fp32", "float32"):
        return torch.float32
    if n in ("f16", "fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype '{name}'. Use f32 or f16.")


def _assert_close(a: torch.Tensor, b: torch.Tensor, name: str, atol: float, rtol: float):
    ok = torch.allclose(a, b, atol=atol, rtol=rtol)
    if not ok:
        max_abs = (a - b).abs().max().item()
        raise AssertionError(f"[FAIL] {name}: allclose failed (max_abs={max_abs}, atol={atol}, rtol={rtol})")
    print(f"[OK] {name}")


def make_fixed_problem(
    Bsz: int = 256,
    Din: int = 1024,
    Dout: int = 10,
    seed: int = 0,
    noise: float = 0.0,
    device: str = "cuda",
    dtype: torch.dtype = torch.float32,
):
    torch.manual_seed(seed)
    W_star = (torch.randn(Din, Dout, device=device, dtype=dtype) * 0.1).contiguous()
    x = torch.randn(Bsz, Din, device=device, dtype=dtype).contiguous()
    t = (x @ W_star).contiguous()
    if noise != 0.0:
        t = (t + noise * torch.randn_like(t)).contiguous()
    return x, t


def make_dataloader(x_fixed: torch.Tensor, t_fixed: torch.Tensor):
    while True:
        yield (Tensor(x_fixed), Tensor(t_fixed))


@torch.no_grad()
def unit_test_relu(dtype: torch.dtype, device: str):
    N = 4096
    x = torch.randn(N, device=device, dtype=dtype).contiguous()
    m = ReLU()
    y = m(Tensor(x)).torch()

    ref = torch.relu(x)
    atol = 1e-5 if dtype == torch.float32 else 5e-3
    rtol = 1e-5 if dtype == torch.float32 else 5e-3
    _assert_close(y, ref, f"relu dtype={dtype}", atol, rtol)


@torch.no_grad()
def unit_test_linear_forward(dtype: torch.dtype, device: str):
    """
    This indirectly validates:
      - GEMM path (f32 naive or f16 TC)
      - BiasAdd path (f32/f16/half2 if you implemented)
    """
    Bsz, Din, Dout = 128, 256, 64
    x = torch.randn(Bsz, Din, device=device, dtype=dtype).contiguous()

    lin = Linear(Din, Dout)
    # Forward through AICF
    y = lin(Tensor(x)).torch()

    # Torch reference: need lin's weights/bias as torch tensors.
    # Assumes Linear exposes .weight/.bias as Tensor or has .torch() access.
    W = lin.weight.torch()
    b = lin.bias.torch()
    ref = (x @ W) + b

    # for f16, compare in fp32
    if dtype == torch.float16:
        _assert_close(y.float(), ref.float(), f"linear_fwd dtype={dtype}", atol=5e-2, rtol=5e-2)
    else:
        _assert_close(y, ref, f"linear_fwd dtype={dtype}", atol=1e-4, rtol=1e-4)


def unit_test_backward_path(dtype: torch.dtype, device: str):
    """
    This stresses:
      - MSE grad (f32/f16)
      - ReLU bwd (f32/f16/half2 if you implemented)
      - ReduceSum (bias grad) (f32 or f16->f32 variant)
      - SGD step (f32/f16/half2)
    We only assert it runs + loss decreases (light sanity).
    """
    torch.manual_seed(0)
    Bsz, Din, H, Dout = 128, 256, 128, 32
    x, t = make_fixed_problem(Bsz=Bsz, Din=Din, Dout=Dout, seed=0, noise=0.0, device=device, dtype=dtype)

    model = Sequential(
        Linear(Din, H),
        ReLU(),
        Linear(H, Dout),
    )
    loss_fn = MSE()
    opt = SGD(lr=float(_env("AICF_LR", "1e-3")))

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        cfg=TrainerConfig(mode="bench", log_every=999999),
    )

    dl = make_dataloader(x, t)

    # run a few steps and check loss trend
    losses = []
    for _ in range(5):
        loss = trainer.step(next(dl))  # if Trainer exposes step; if not, fallback below
        # Some trainers return float; some return Tensor. Handle both.
        if isinstance(loss, Tensor):
            losses.append(float(loss.torch().item()))
        else:
            losses.append(float(loss))

    if not (losses[-1] <= losses[0] + 1e-6):
        raise AssertionError(f"[FAIL] backward_path: loss did not decrease: {losses}")
    print(f"[OK] backward_path dtype={dtype} loss {losses[0]:.6f} -> {losses[-1]:.6f}")


def run_train(dtype: torch.dtype, device: str):
    steps = int(_env("AICF_STEPS", "100"))
    log_every = int(_env("AICF_LOG_EVERY", "10"))
    lr = float(_env("AICF_LR", "1e-3"))
    seed = int(_env("AICF_SEED", "0"))
    noise = float(_env("AICF_NOISE", "0.0"))

    Bsz = int(_env("AICF_B", "256"))
    Din = int(_env("AICF_DIN", "1024"))
    H = int(_env("AICF_H", "256"))
    Dout = int(_env("AICF_DOUT", "10"))

    x_fixed, t_fixed = make_fixed_problem(
        Bsz=Bsz, Din=Din, Dout=Dout, seed=seed, noise=noise, device=device, dtype=dtype
    )

    model = Sequential(
        Linear(Din, H),
        ReLU(),
        Linear(H, Dout),
    )

    loss_fn = MSE()
    opt = SGD(lr=lr)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        cfg=TrainerConfig(mode="bench", log_every=log_every),
    )

    dl = make_dataloader(x_fixed, t_fixed)
    trainer.fit(dl, steps=steps)


def main():
    os.environ.setdefault("AICF_BACKEND", "aicf")

    print("backend path =", B.__file__)
    print("AICF_BACKEND =", os.environ.get("AICF_BACKEND"))
    print("PYMOD_DIR =", PYMOD_DIR)

    assert torch.cuda.is_available(), "CUDA required for this example"
    device = "cuda"

    dtype = _torch_dtype(_env("AICF_DTYPE", "f32"))
    run_units = int(_env("AICF_UNIT_TESTS", "1")) != 0

    if run_units:
        print(f"\n== Unit tests (dtype={dtype}) ==")
        unit_test_relu(dtype, device)
        unit_test_linear_forward(dtype, device)

        # Backward-path stress test (optional)
        if int(_env("AICF_UNIT_BWD", "1")) != 0:
            unit_test_backward_path(dtype, device)

    print(f"\n== Train bench (dtype={dtype}) ==")
    run_train(dtype, device)


if __name__ == "__main__":
    main()
