from __future__ import annotations

import torch


class Optimizer:
    """
    Base optimizer interface for AICF.
    Requirements:
      - state tensors must be real device tensors so CUDA Graph capture sees them.
      - meta update should be done via in-place tensor ops (fill_/copy_) not Python scalars.
    """

    def named_state_tensors(self) -> dict[str, torch.Tensor]:
        """
        Return ALL persistent state tensors that must be captured/replayed deterministically.
        Example (Adam):
          {
            "step": step_tensor,
            "m/0.W": m_tensor_for_param,
            "v/0.W": v_tensor_for_param,
            ...
            "bc1_inv": bc1_inv_tensor,
            "bc2_inv": bc2_inv_tensor,
          }
        """
        raise NotImplementedError

    def named_meta_tensors(self) -> dict[str, torch.Tensor]:
        """
        Return meta tensors that may be host-updated between replays.
        These must still be device tensors (or at least tensors captured by graph).
        Example (Adam):
          {"bc1_inv": bc1_inv, "bc2_inv": bc2_inv, "lr": lr_tensor?}
        Default: no meta tensors.
        """
        return {}

    def update_meta(self):
        """
        Host-managed meta update (must be in-place tensor updates, e.g., fill_/copy_).
        Called on each train step (eager or compiled) to advance bias-correction etc.
        """
        raise NotImplementedError
