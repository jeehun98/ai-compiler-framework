# aicf_fw/optim/sgd.py
from __future__ import annotations
from typing import List, Optional

import torch

from .base import Optimizer
from ..core.tensor import Tensor
from ..backend import get_backend


class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 1e-4, inplace: bool = True,
                 grad_clip: Optional[float] = None):
        super().__init__(params)
        self.lr = float(lr)
        self.inplace = bool(inplace)
        self.grad_clip = grad_clip  # elementwise clip (e.g., 1.0)

    def step(self) -> None:
        backend = get_backend()

        for p in self.params:
            if p.grad is None:
                continue

            P = p.data
            G = p.grad.data

            if P is None or G is None:
                continue

            # --- guard: grad finite ---
            if not torch.isfinite(G).all():
                continue

            # --- optional grad clipping (elementwise) ---
            if self.grad_clip is not None:
                c = float(self.grad_clip)
                G_use = G.clamp(min=-c, max=c)
            else:
                G_use = G

            attrs = {"lr": self.lr}

            if self.inplace:
                # ✅ output = param itself (matches current C++: O -= lr * G)
                backend.op_call_out("sgd_step", [P, G_use], [P], attrs)
            else:
                # out-of-place는 현재 C++ 구현과 의미가 다르므로 금지
                raise RuntimeError(
                    "SGD(inplace=False) is not supported with current SgdStep kernel "
                    "(kernel updates output in-place and does not read input param)."
                )
