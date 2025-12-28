# aicf_fw/optim/sgd.py
from __future__ import annotations

from typing import Any, Dict, Optional, Union, Iterable

import torch

from aicf_fw.core.tensor import Tensor
from aicf_fw.core.module import Module
from aicf_fw.backend import get_backend
from .base import Optimizer, ParamsLike


class SGD(Optimizer):
    """
    SGD optimizer.
    - params: Module or iterable[Tensor]
    - inplace=True uses backend.op_call_out("sgd_step", ...) for in-place update
    """
    def __init__(
        self,
        params: ParamsLike,
        lr: float = 1e-3,
        inplace: bool = True,
        grad_clip: Optional[float] = None,
    ):
        super().__init__(params)
        self.lr = float(lr)
        self.inplace = bool(inplace)
        self.grad_clip = float(grad_clip) if grad_clip is not None else None

    def step(self) -> None:
        backend = get_backend()

        for p in self.params:
            if not p.requires_grad:
                continue
            if p.grad is None:
                continue

            w = p.data
            g = p.grad.data

            # optional grad clip (torch-side, cheap and safe)
            if self.grad_clip is not None:
                # clip by value (not norm) - simple
                g = torch.clamp(g, min=-self.grad_clip, max=self.grad_clip)

            if self.inplace:
                # in-place update: w <- w - lr*g
                backend.op_call_out(
                    "sgd_step",
                    [w, g],
                    [w],
                    {"lr": self.lr},
                )
            else:
                # out-of-place fallback (torch)
                p.data = w - self.lr * g

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": "SGD",
            "lr": self.lr,
            "inplace": self.inplace,
            "grad_clip": self.grad_clip,
            "param_count": len(self.params),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # minimal strict-ish load
        if state.get("type") != "SGD":
            raise ValueError(f"state_dict type mismatch: {state.get('type')}")
        self.lr = float(state.get("lr", self.lr))
        self.inplace = bool(state.get("inplace", self.inplace))
        gc = state.get("grad_clip", self.grad_clip)
        self.grad_clip = float(gc) if gc is not None else None
