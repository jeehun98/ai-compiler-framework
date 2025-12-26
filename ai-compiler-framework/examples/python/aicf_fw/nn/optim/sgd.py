# aicf_fw/nn/optim/sgd.py
from __future__ import annotations
from typing import Iterable
from ...core.tensor import Parameter, Tensor
from ...backend import get_backend
from ...core.autograd import no_grad

class SGD:
    def __init__(self, params: Iterable[Parameter], lr: float):
        self.params = list(params)
        self.lr = lr

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()

    def step(self):
        backend = get_backend()
        # updates should not be recorded in autograd graph
        with no_grad():
            for p in self.params:
                if p.grad is None:
                    continue
                new_data = backend.op_call("sgd_step", [p.data, p.grad.data], {"lr": self.lr})
                p.data = new_data
