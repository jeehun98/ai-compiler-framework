# examples/python/aicf_fw/tensor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional
import torch

_DTYPE_STR = {
    torch.float16: "float16",
    torch.float32: "float32",
}

_STR_DTYPE = {
    "float16": torch.float16,
    "float32": torch.float32,
}


@dataclass
class Tensor:
    """
    v0.2:
    - torch.Tensor를 직접 보관 (캡처/리플레이/pybind op_call에 필요)
    - shape/stride/dtype/device는 torch.Tensor의 view
    """
    t: torch.Tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.t.shape)

    @property
    def stride(self) -> Tuple[int, ...]:
        return tuple(self.t.stride())

    @property
    def dtype(self) -> str:
        return _DTYPE_STR.get(self.t.dtype, str(self.t.dtype))

    @property
    def device(self) -> str:
        return "cuda" if self.t.is_cuda else "cpu"

    @property
    def data_ptr(self) -> int:
        return int(self.t.data_ptr())

    def contiguous(self) -> "Tensor":
        return Tensor(self.t.contiguous())

    def view(self, *shape: int) -> "Tensor":
        return Tensor(self.t.view(*shape))

    def reshape(self, *shape: int) -> "Tensor":
        return Tensor(self.t.reshape(*shape))

    def transpose(self, dim0: int, dim1: int) -> "Tensor":
        return Tensor(self.t.transpose(dim0, dim1))

    def t_(self) -> "Tensor":
        # 2D transpose convenience
        if self.t.ndim != 2:
            raise ValueError("t_() requires 2D tensor")
        return Tensor(self.t.t())

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype}, device={self.device}, contig={self.t.is_contiguous()})"


@dataclass
class Parameter:
    data: Tensor
