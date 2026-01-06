# aicf_fw/core/tensor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch


@dataclass
class TensorMeta:
    shape: Tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


class Tensor:
    """
    data: backend handle (torch.Tensor or None for symbolic)
    creator: Node that created this Tensor (None for leaf/Parameter)
    grad: Tensor or backend handle (we keep as Tensor for simplicity)
    meta: TensorMeta (always available; for symbolic tensors meta is required)
    """
    __slots__ = ("data", "requires_grad", "grad", "creator", "name", "meta")

    def __init__(
        self,
        data: Any,
        requires_grad: bool = False,
        creator=None,
        name: str = "",
        meta: Optional[TensorMeta] = None,
    ):
        self.data = data
        self.requires_grad = requires_grad
        self.grad: Optional["Tensor"] = None
        self.creator = creator
        self.name = name

        if meta is not None:
            self.meta = meta
        else:
            if isinstance(data, torch.Tensor):
                self.meta = TensorMeta(tuple(data.shape), data.dtype, data.device)
            else:
                raise TypeError("Tensor(meta=None) requires torch.Tensor data; for symbolic Tensor, pass meta.")

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.meta.shape)

    @property
    def dtype(self) -> torch.dtype:
        return self.meta.dtype

    @property
    def device(self) -> torch.device:
        return self.meta.device

    @property
    def is_symbolic(self) -> bool:
        return self.data is None

    def zero_grad(self):
        self.grad = None

    def backward(self, grad: Optional["Tensor"] = None):
        from .autograd import backward as autograd_backward
        autograd_backward(self, grad)


class Parameter(Tensor):
    def __init__(self, data: Any, requires_grad: bool = True, name: str = ""):
        super().__init__(data, requires_grad=requires_grad, creator=None, name=name)
