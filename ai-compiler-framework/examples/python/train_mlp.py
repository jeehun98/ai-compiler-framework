from __future__ import annotations

import os
import sys
from pathlib import Path

# --- AICF build/python bootstrap (for aicf_cuda) ---
REPO_ROOT = Path(__file__).resolve().parents[2]  # repo/examples/python/train_mlp.py -> repo
PYMOD_DIR = REPO_ROOT / "build" / "python"
PKG_DIR   = PYMOD_DIR / "aicf_cuda"

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


def main():
    os.environ.setdefault("AICF_BACKEND", "aicf")

    print("backend path =", B.__file__)
    print("AICF_BACKEND =", os.environ.get("AICF_BACKEND"))
    print("PYMOD_DIR =", PYMOD_DIR)

    assert torch.cuda.is_available(), "CUDA required for this example"
    device = "cuda"

    steps = int(os.environ.get("AICF_STEPS", "100"))
    log_every = int(os.environ.get("AICF_LOG_EVERY", "10"))
    lr = float(os.environ.get("AICF_LR", "1e-3"))
    seed = int(os.environ.get("AICF_SEED", "0"))
    noise = float(os.environ.get("AICF_NOISE", "0.0"))

    Bsz = int(os.environ.get("AICF_B", "256"))
    Din = int(os.environ.get("AICF_DIN", "1024"))
    H = int(os.environ.get("AICF_H", "256"))
    Dout = int(os.environ.get("AICF_DOUT", "10"))

    x_fixed, t_fixed = make_fixed_problem(
        Bsz=Bsz, Din=Din, Dout=Dout, seed=seed, noise=noise, device=device, dtype=torch.float32
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


if __name__ == "__main__":
    main()
