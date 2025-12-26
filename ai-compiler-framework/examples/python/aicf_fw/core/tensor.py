# aicf_fw/core/tensor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional

class Tensor:
    """
    data: backend handle (torch.Tensor or aicf handle)
    creator: Node that created this Tensor (None for leaf/Parameter)
    grad: Tensor or backend handle (we keep as Tensor for simplicity)
    """
    __slots__ = ("data", "requires_grad", "grad", "creator", "name")

    def __init__(self, data: Any, requires_grad: bool = False, creator=None, name: str = ""):
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional["Tensor"] = None
        self.creator = creator
        self.name = name

    def zero_grad(self):
        self.grad = None

    def backward(self, grad: Optional["Tensor"] = None):
        # Late import to avoid cycles
        from .autograd import backward as autograd_backward
        autograd_backward(self, grad)

class Parameter(Tensor):
    def __init__(self, data: Any, requires_grad: bool = True, name: str = ""):
        super().__init__(data, requires_grad=requires_grad, creator=None, name=name)
