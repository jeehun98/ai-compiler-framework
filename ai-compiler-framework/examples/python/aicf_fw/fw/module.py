from __future__ import annotations
from collections import OrderedDict
import torch
from aicf_fw.fw.naming import param_name

class Module:
    def __init__(self):
        self._modules: OrderedDict[str, Module] = OrderedDict()
        self._params: OrderedDict[str, torch.Tensor] = OrderedDict()
        self._prefix: str = ""  # set by parent

    # ----- registration -----
    def add_module(self, name: str, m: "Module"):
        self._modules[name] = m
        m._prefix = name if self._prefix == "" else f"{self._prefix}.{name}"

    def register_parameter(self, local_name: str, t: torch.Tensor):
        self._params[local_name] = t

    # ----- query -----
    def named_parameters(self, prefix: str | None = None):
        pfx = self._prefix if prefix is None else prefix

        for local, t in self._params.items():
            yield param_name(pfx, local), t

        for child_name, m in self._modules.items():
            child_prefix = child_name if pfx == "" else f"{pfx}.{child_name}"
            yield from m.named_parameters(prefix=child_prefix)

    def parameters(self):
        for _, p in self.named_parameters():
            yield p

    # ----- device/dtype -----
    def to(self, device: str | torch.device, dtype: torch.dtype | None = None):
        dev = torch.device(device) if isinstance(device, str) else device
        for k, t in list(self._params.items()):
            self._params[k] = t.to(device=dev, dtype=(dtype if dtype is not None else t.dtype))
        for _, m in self._modules.items():
            m.to(dev, dtype=dtype)
        return self

    # ----- IR forward -----
    def forward_ir(self, x_sym, psym: dict[str, object]):
        raise NotImplementedError
