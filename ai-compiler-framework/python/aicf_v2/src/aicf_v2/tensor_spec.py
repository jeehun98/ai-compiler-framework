from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class TensorSpec:
    shape: Tuple[int, ...]
    dtype: str   # "f16", "f32", "i32", ...
    device: str  # "cuda", "cpu"

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))
        if len(self.shape) == 0:
            raise ValueError("TensorSpec.shape must be non-empty")
