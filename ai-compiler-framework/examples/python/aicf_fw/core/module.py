# aicf_fw/core/module.py
from __future__ import annotations
from typing import Dict, Iterator, Tuple, Any, List
from .tensor import Parameter, Tensor

class Module:
    def __init__(self):
        self.training = True
        self._parameters: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}

    def train(self):
        self.training = True
        for m in self._modules.values():
            m.train()
        return self

    def eval(self):
        self.training = False
        for m in self._modules.values():
            m.eval()
        return self

    def parameters(self) -> Iterator[Parameter]:
        for _, p in self.named_parameters():
            yield p

    def named_parameters(self, prefix: str = "") -> Iterator[Tuple[str, Parameter]]:
        for name, p in self._parameters.items():
            key = f"{prefix}{name}" if prefix == "" else f"{prefix}.{name}"
            yield key, p
        for name, m in self._modules.items():
            subprefix = f"{prefix}{name}" if prefix == "" else f"{prefix}.{name}"
            yield from m.named_parameters(subprefix)

    def add_parameter(self, name: str, p: Parameter):
        self._parameters[name] = p

    def add_module(self, name: str, m: "Module"):
        self._modules[name] = m

    def __setattr__(self, name: str, value: Any):
        # auto-register Parameter/Module
        if isinstance(value, Parameter):
            if "_parameters" in self.__dict__:
                self._parameters[name] = value
        if isinstance(value, Module):
            if "_modules" in self.__dict__:
                self._modules[name] = value
        super().__setattr__(name, value)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError
