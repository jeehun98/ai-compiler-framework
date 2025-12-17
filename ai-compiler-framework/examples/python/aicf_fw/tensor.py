# examples/python/aicf_fw/tensor.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class Tensor:
    t: torch.Tensor

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.t.shape)

    @property
    def requires_grad(self) -> bool:
        return bool(self.t.requires_grad)

    @property
    def grad(self) -> Optional["Tensor"]:
        if self.t.grad is None:
            return None
        return Tensor(self.t.grad)

    @staticmethod
    def empty(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[str] = None, dtype=torch.float32) -> "Tensor":
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return Tensor(torch.empty(shape, device=dev, dtype=dtype, requires_grad=requires_grad))

    @staticmethod
    def randn(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[str] = None, dtype=torch.float32) -> "Tensor":
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return Tensor(torch.randn(shape, device=dev, dtype=dtype, requires_grad=requires_grad))

    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False, device: Optional[str] = None, dtype=torch.float32) -> "Tensor":
        dev = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return Tensor(torch.zeros(shape, device=dev, dtype=dtype, requires_grad=requires_grad))

    def signature(self) -> str:
        return f"{tuple(self.t.shape)}:{str(self.t.dtype).replace('torch.', '')}:{str(self.t.device)}"

    # ✅ 안전한 in-place: leaf 파라미터에도 쓰기 쉽게 no_grad로 감싼 버전 제공
    def zero_(self) -> None:
        with torch.no_grad():
            self.t.zero_()

    def add_(self, other: "Tensor", alpha: float = 1.0) -> None:
        with torch.no_grad():
            self.t.add_(other.t, alpha=alpha)

    # 필요하면 raw in-place도 분리 제공 (그래야 autograd 그래프 안에서 in-place가 필요한 경우만 쓴다)
    def add_in_graph_(self, other: "Tensor", alpha: float = 1.0) -> None:
        self.t.add_(other.t, alpha=alpha)

    def backward(self) -> None:
        self.t.backward()
