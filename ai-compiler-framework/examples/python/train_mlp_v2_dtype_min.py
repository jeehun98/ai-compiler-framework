from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PYMOD_DIR = REPO_ROOT / "build" / "python"
PKG_DIR   = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(PYMOD_DIR))

if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import torch

from aicf_fw.tensor import Tensor
from aicf_fw.modules.sequential import Sequential
from aicf_fw.modules.linear import Linear
from aicf_fw.modules.activations import ReLU
from aicf_fw.losses.mse import MSE
from aicf_fw.optim.sgd import SGD
from aicf_fw.trainer import Trainer, TrainerConfig
from aicf_fw.modules.relu import ReLU
import aicf_fw.backend as B


def parse_dtype_env() -> torch.dtype:
    s = os.environ.get("AICF_DTYPE", "f32").lower()
    if s in ("f16", "fp16", "half"):
        return torch.float16
    return torch.float32


def make_fixed_problem(Bsz=256, Din=1024, Dout=10, seed=0, noise=0.0, device="cuda", dtype=torch.float32):
    torch.manual_seed(seed)
    W_star = (torch.randn(Din, Dout, device=device, dtype=torch.float32) * 0.1).contiguous()
    x = torch.randn(Bsz, Din, device=device, dtype=dtype).contiguous()
    t = (x.float() @ W_star).to(dtype=dtype).contiguous()
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

    assert torch.cuda.is_available()
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

    dtype = parse_dtype_env()
    print("ENV AICF_DTYPE =", os.environ.get("AICF_DTYPE"))
    print("DTYPE =", dtype)

    x_fixed, t_fixed = make_fixed_problem(Bsz=Bsz, Din=Din, Dout=Dout, seed=seed, noise=noise, device=device, dtype=dtype)
    print("x_fixed dtype =", x_fixed.dtype)
    print("t_fixed dtype =", t_fixed.dtype)

    # parameters are f32 (by Linear module policy)
    model = Sequential(
        Linear(Din, H, bias=False),
        ReLU(),
        Linear(H, Dout, bias=False),
    )
    print("Linear class file =", model.modules[0].__class__.__module__, model.modules[0].__class__)
    print("Linear forward =", model.modules[0].__class__.forward)
    print("Linear MRO =", [c.__name__ for c in model.modules[0].__class__.mro()])


    loss_fn = MSE()
    opt = SGD(lr=lr)

    # AFTER
    trainer = Trainer(
        model=model,
        optim=opt,
        cfg=TrainerConfig(mode="bench"),
    )

    dl = make_dataloader(x_fixed, t_fixed)
    trainer.fit(dl, steps=steps)


if __name__ == "__main__":
    main()


# $env:AICF_BACKEND="aicf"
# $env:AICF_DTYPE="f16" 
# $env:AICF_LR="1e-3"
# python train_mlp_v2_dtype_min.py



