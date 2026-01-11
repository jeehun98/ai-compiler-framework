from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

from aicf_fw.core.autograd import Tensor
from aicf_fw.core.module import Module

ParamsLike = Union[Module, Iterable[Tensor]]


class Optimizer:
    def __init__(self, params: ParamsLike):
        if isinstance(params, Module):
            self.model: Optional[Module] = params
            self.params: List[Tensor] = list(params.parameters())
        else:
            self.model = None
            self.params = list(params)

        for p in self.params:
            if not isinstance(p, Tensor):
                raise TypeError(f"Optimizer expects Tensor params, got {type(p)}")

    def zero_grad(self, set_to_none: bool = False) -> None:
        """
        Capture-safe default: keep grad buffers, do NOT set to None.
        We intentionally do NOT call self.model.zero_grad because it might set grad=None.
        """
        for p in self.params:
            if not p.requires_grad:
                continue
            if set_to_none:
                p.grad = None
            else:
                # keep buffer; do NOT torch.zero_ here (avoid default-stream ops inside capture)
                # overwrite-mode backward will refresh grads each step.
                pass

    def step(self) -> None:
        raise NotImplementedError

    def state_dict(self) -> Dict[str, Any]:
        return {"param_count": len(self.params)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        _ = state
