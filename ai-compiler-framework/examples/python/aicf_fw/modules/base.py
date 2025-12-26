# examples/python/aicf_fw/modules/base.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, Optional

from ..tensor import Tensor


@dataclass
class Parameter:
    data: Tensor

    @property
    def grad(self) -> Optional[Tensor]:
        # torch autograd를 쓰든, AICF manual backward를 쓰든
        # grad는 결국 data.t.grad에 저장된다.
        return self.data.grad

    def zero_grad(self) -> None:
        # ✅ 가장 안전: 누적을 끊기 위해 grad를 None으로
        if self.data.t.grad is not None:
            self.data.t.grad = None


class Module:
    def __init__(self) -> None:
        self._params: Dict[str, Parameter] = {}
        self._modules: Dict[str, "Module"] = {}

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def add_parameter(self, name: str, p: Parameter) -> None:
        self._params[name] = p

    def add_module(self, name: str, m: "Module") -> None:
        self._modules[name] = m

    def parameters(self) -> Iterator[Parameter]:
        yield from self._params.values()
        for m in self._modules.values():
            yield from m.parameters()

    def zero_grad(self) -> None:
        for p in self.parameters():
            p.zero_grad()
