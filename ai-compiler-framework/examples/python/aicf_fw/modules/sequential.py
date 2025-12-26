# examples/python/aicf_fw/modules/sequential.py
from __future__ import annotations
from .base import Module

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = list(modules)

    def forward(self, x):
        for i, m in enumerate(self.modules):
            try:
                x = m(x)
            except NotImplementedError as e:
                raise NotImplementedError(
                    f"Sequential: module[{i}]={m.__class__.__name__} has no forward()"
                ) from e
        return x
