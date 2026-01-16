from __future__ import annotations
import torch

class Optimizer:
    def named_state_tensors(self) -> dict[str, torch.Tensor]:
        raise NotImplementedError

    def update_meta(self):
        # host-managed meta 업데이트 (fill_)
        raise NotImplementedError
