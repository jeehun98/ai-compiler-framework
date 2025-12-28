# aicf_fw/optim/base.py
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Union

from aicf_fw.core.tensor import Tensor
from aicf_fw.core.module import Module


ParamsLike = Union[Module, Iterable[Tensor]]


class Optimizer:
    def __init__(self, params: ParamsLike):
        # params can be a Module or an iterable of Tensor
        if isinstance(params, Module):
            self.model: Optional[Module] = params
            self.params: List[Tensor] = list(params.parameters())
        else:
            self.model = None
            self.params = list(params)

        # sanity
        for p in self.params:
            if not isinstance(p, Tensor):
                raise TypeError(f"Optimizer expects Tensor params, got {type(p)}")

    def zero_grad(self, set_to_none: bool = True) -> None:
        if self.model is not None:
            self.model.zero_grad(set_to_none=set_to_none)
            return
        for p in self.params:
            if set_to_none:
                p.grad = None
            else:
                p.grad = None

    def step(self) -> None:
        raise NotImplementedError

    # ----------------------------
    # checkpointing
    # ----------------------------
    def state_dict(self) -> Dict[str, Any]:
        # minimal: store hyperparams + (optional) per-param state keyed by index
        return {"param_count": len(self.params)}

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        # minimal: no-op unless derived optimizer overrides
        _ = state
