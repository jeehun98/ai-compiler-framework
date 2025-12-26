from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import os
import torch

from .base import Optimizer
from ..modules.base import Parameter


@dataclass
class SGD(Optimizer):
    lr: float = 1e-2

    def step(self, params: Iterable[Parameter]) -> None:
        kind = os.environ.get("AICF_BACKEND", "torch").lower()

        # ---- AICF path ----
        if kind == "aicf":
            from aicf_cuda import _C

            for p in params:
                w = p.data.t
                g = w.grad  # <- PyTorch autograd grad

                if g is None:
                    continue

                # capture-safe: no allocations here
                if (not w.is_contiguous()) or (not g.is_contiguous()):
                    raise RuntimeError(
                        "AICF SGDStep requires contiguous W and dW. "
                        "Ensure parameters and grads are contiguous before capture."
                    )

                _C.op_call(_C.OpKind.SgdStep, [w, g], [w], {"lr": float(self.lr)})

            return

        # ---- Torch fallback (reference) ----
        with torch.no_grad():
            for p in params:
                w = p.data.t
                g = w.grad
                if g is None:
                    continue
                w.add_(g, alpha=-self.lr)
