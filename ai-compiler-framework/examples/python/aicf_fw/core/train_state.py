# aicf_fw/python_framework_test/train_state.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


@dataclass
class TrainState:
    """
    Full training state snapshot for determinism and restore checks.

    Captures:
      - model parameters (Tensor.data as torch.Tensor)
      - parameter grads (Tensor.grad.data or None)
      - Adam moments (optim.m / optim.v) keyed by internal param index
      - Adam step, bc1_inv, bc2_inv
    """
    params: Dict[str, torch.Tensor]
    grads: Dict[str, Optional[torch.Tensor]]
    adam_m: Dict[int, torch.Tensor]
    adam_v: Dict[int, torch.Tensor]
    step: torch.Tensor
    bc1_inv: torch.Tensor
    bc2_inv: torch.Tensor

    @staticmethod
    @torch.no_grad()
    def capture(model: Any, optim: Any) -> "TrainState":
        ps: Dict[str, torch.Tensor] = {n: p.data.detach().clone() for n, p in model.named_parameters()}
        gs: Dict[str, Optional[torch.Tensor]] = {}
        for n, p in model.named_parameters():
            if getattr(p, "grad", None) is None:
                gs[n] = None
            else:
                gs[n] = p.grad.data.detach().clone()

        ms: Dict[int, torch.Tensor] = {i: optim.m[i].data.detach().clone() for i in optim.m.keys()}
        vs: Dict[int, torch.Tensor] = {i: optim.v[i].data.detach().clone() for i in optim.v.keys()}

        step = optim.step.detach().clone()
        bc1 = optim.bc1_inv.detach().clone()
        bc2 = optim.bc2_inv.detach().clone()
        return TrainState(ps, gs, ms, vs, step, bc1, bc2)

    @torch.no_grad()
    def restore(self, model: Any, optim: Any) -> None:
        # Lazy import to avoid circular deps at import time
        from aicf_fw.core.autograd import Tensor  # noqa: WPS433

        cur = {n: p for n, p in model.named_parameters()}

        for n, src in self.params.items():
            cur[n].data.copy_(src)

        for n, g in self.grads.items():
            p = cur[n]
            if g is None:
                p.grad = None
            else:
                if getattr(p, "grad", None) is None:
                    p.grad = Tensor(torch.empty_like(g), requires_grad=False)
                p.grad.data.copy_(g)

        for i in self.adam_m.keys():
            optim.m[i].data.copy_(self.adam_m[i])
            optim.v[i].data.copy_(self.adam_v[i])

        optim.step.copy_(self.step)
        optim.bc1_inv.copy_(self.bc1_inv)
        optim.bc2_inv.copy_(self.bc2_inv)

    @torch.no_grad()
    def assert_equal(self, model: Any, optim: Any, *, tag: str = "") -> None:
        cur = {n: p for n, p in model.named_parameters()}

        for n, ref in self.params.items():
            d = _max_abs_diff(cur[n].data, ref)
            if d != 0.0:
                raise AssertionError(f"[state] param mismatch {tag}: {n} maxdiff={d}")

        for n, refg in self.grads.items():
            pg = getattr(cur[n], "grad", None)
            if refg is None:
                if pg is not None:
                    raise AssertionError(f"[state] grad mismatch {tag}: {n} expected None")
            else:
                if pg is None:
                    raise AssertionError(f"[state] grad mismatch {tag}: {n} expected tensor, got None")
                d = _max_abs_diff(pg.data, refg)
                if d != 0.0:
                    raise AssertionError(f"[state] grad mismatch {tag}: {n} maxdiff={d}")

        for i in self.adam_m.keys():
            dm = _max_abs_diff(optim.m[i].data, self.adam_m[i])
            dv = _max_abs_diff(optim.v[i].data, self.adam_v[i])
            if dm != 0.0:
                raise AssertionError(f"[state] m mismatch {tag}: idx={i} maxdiff={dm}")
            if dv != 0.0:
                raise AssertionError(f"[state] v mismatch {tag}: idx={i} maxdiff={dv}")

        if int(optim.step.item()) != int(self.step.item()):
            raise AssertionError(
                f"[state] step mismatch {tag}: {int(optim.step.item())} != {int(self.step.item())}"
            )
        if float(optim.bc1_inv.item()) != float(self.bc1_inv.item()):
            raise AssertionError(
                f"[state] bc1 mismatch {tag}: {optim.bc1_inv.item()} != {self.bc1_inv.item()}"
            )
        if float(optim.bc2_inv.item()) != float(self.bc2_inv.item()):
            raise AssertionError(
                f"[state] bc2 mismatch {tag}: {optim.bc2_inv.item()} != {self.bc2_inv.item()}"
            )
