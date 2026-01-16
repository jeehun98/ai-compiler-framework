# aicf_fw/optim/base.py
from __future__ import annotations
import torch


class Optimizer:
    """
    Base class for core_v2 engine optimizers.

    Contract:
      - named_state_tensors(): returns {name: torch.Tensor}
        -> optimizer state + meta tensors to be bound as statics/params
      - update_meta(): updates host-managed meta tensors (fill_)
    """

    def named_state_tensors(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def update_meta(self):
        raise NotImplementedError
