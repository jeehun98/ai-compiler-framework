# aicf_fw/optim/sgd.py
from __future__ import annotations

from typing import Any, Dict, Optional

from aicf_fw.backend import get_backend
from .base import Optimizer, ParamsLike


class SGD(Optimizer):
    """
    SGD optimizer (capture-safe flavor).

    Policy:
    - inplace=True uses backend.op_call_out("sgd_step") to keep pointer stability.
    - DO NOT use torch-side ops (like clamp) in the capture path.
    - With autograd overwrite mode, you usually don't need zero_grad at all.
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

        # For capture safety, grad_clip via torch is not allowed inside capture.
        # If you need grad_clip, implement an AICF op (clip) and call it in the graph.
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

            if self.grad_clip is not None:
                # Capture-safe note:
                # torch.clamp here would introduce torch ops.
                # If you really want clipping, add an AICF op "clip" and use op_call_out.
                raise RuntimeError(
                    "SGD.grad_clip is set, but torch-side clamp is not capture-safe. "
                    "Implement AICF clip op or set grad_clip=None."
                )

            if self.inplace:
                backend.op_call_out(
                    "sgd_step",
                    [w, g],
                    [w],
                    {"lr": self.lr},
                )
            else:
                # out-of-place update is NOT capture-safe (rebinds tensor storage)
                raise RuntimeError("SGD(inplace=False) is not capture-safe. Use inplace=True.")

    def state_dict(self) -> Dict[str, Any]:
        return {
            "type": "SGD",
            "lr": self.lr,
            "inplace": self.inplace,
            "grad_clip": self.grad_clip,
            "param_count": len(self.params),
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        if state.get("type") != "SGD":
            raise ValueError(f"state_dict type mismatch: {state.get('type')}")
        self.lr = float(state.get("lr", self.lr))
        self.inplace = bool(state.get("inplace", self.inplace))
        gc = state.get("grad_clip", self.grad_clip)
        self.grad_clip = float(gc) if gc is not None else None
