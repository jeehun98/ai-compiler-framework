# examples/python/train_mlp.py
from __future__ import annotations
import itertools

from aicf_fw.tensor import Tensor
from aicf_fw.modules.sequential import Sequential
from aicf_fw.modules.linear import Linear
from aicf_fw.modules.activations import ReLU
from aicf_fw.losses.mse import MSELoss
from aicf_fw.optim.sgd import SGD
from aicf_fw.trainer import Trainer, TrainerConfig
import aicf_fw.backend as B

import torch
import os

print("backend path =", B.__file__)

from aicf_fw.tensor import Tensor


# 고정된 ground-truth (매 실행마다 동일하게 하고 싶으면 manual_seed 추가)
torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"

W_star = torch.randn(1024, 10, device=device) * 0.1
x_fixed = torch.randn(256, 1024, device=device)
t_fixed = x_fixed @ W_star  # noise 없이

def make_dataloader():
    from aicf_fw.tensor import Tensor
    while True:
        yield (Tensor(x_fixed), Tensor(t_fixed))

def main():
    model = Sequential(
        Linear(1024, 10),   # ReLU 제거
    )

    loss_fn = MSELoss()
    opt = SGD(lr=1e-2)

    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        optimizer=opt,
        cfg=TrainerConfig(mode="bench", log_every=10),
    )

    dl = make_dataloader()
    steps = int(os.environ.get("AICF_STEPS", "100"))
    trainer.fit(dl, steps=steps)

if __name__ == "__main__":
    main()



# ncu --target-processes all -o aicf_one_step   --section SpeedOfLight --section LaunchStats --section SchedulerStats   "C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe"   "c:\Users\owner\Desktop\ai-compiler-framework\ai-compiler-framework\ai-compiler-framework\examples\python\train_mlp.py"

# ncu --target-processes all -o aicf_step6 --section SpeedOfLight --section LaunchStats --section SchedulerStats   "C:\Users\owner\AppData\Local\Programs\Python\Python312\python.exe"   "c:\Users\owner\Desktop\ai-compiler-framework\ai-compiler-framework\ai-compiler-framework\examples\python\train_mlp.py"