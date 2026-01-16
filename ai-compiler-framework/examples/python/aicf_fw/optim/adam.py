from __future__ import annotations
import torch
from aicf_fw.optim.base import Optimizer
from aicf_fw.fw.naming import opt_m_name, opt_v_name, BC1_NAME, BC2_NAME

class Adam(Optimizer):
    def __init__(
        self,
        model,
        lr: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        device: str | torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        self.lr = float(lr)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.eps = float(eps)

        named_params = list(model.named_parameters())
        assert len(named_params) > 0

        dev = torch.device(device) if device is not None else named_params[0][1].device

        # host-managed meta (pointer-stable scalars)
        self.step_host = 0
        self.bc1_inv = torch.ones((), device=dev, dtype=dtype)
        self.bc2_inv = torch.ones((), device=dev, dtype=dtype)

        # state
        self.m: dict[str, torch.Tensor] = {}
        self.v: dict[str, torch.Tensor] = {}
        for pname, p in named_params:
            self.m[pname] = torch.zeros_like(p)
            self.v[pname] = torch.zeros_like(p)

    def update_meta(self):
        self.step_host += 1
        bc1 = 1.0 / (1.0 - (self.beta1 ** self.step_host))
        bc2 = 1.0 / (1.0 - (self.beta2 ** self.step_host))
        self.bc1_inv.fill_(float(bc1))
        self.bc2_inv.fill_(float(bc2))

    def named_state_tensors(self) -> dict[str, torch.Tensor]:
        d: dict[str, torch.Tensor] = {
            BC1_NAME: self.bc1_inv,
            BC2_NAME: self.bc2_inv,
        }
        for pname in self.m:
            d[opt_m_name(pname)] = self.m[pname]
            d[opt_v_name(pname)] = self.v[pname]
        return d
