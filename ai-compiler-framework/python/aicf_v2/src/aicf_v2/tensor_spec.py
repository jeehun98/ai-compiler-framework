from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple, Iterable

@dataclass(frozen=True)
class TensorSpec:
    shape: Tuple[int, ...]
    dtype: Optional[str] = None   # "f16", "f32", "i32", ...
    device: Optional[str] = None  # "cuda", "cpu"

    def __post_init__(self):
        if not isinstance(self.shape, tuple):
            object.__setattr__(self, "shape", tuple(self.shape))

        # scalar shape 허용해야 AdamStep bc1/bc2가 됨
        # (지금은 len==0 금지라서, 너 AdamStep 넣으면 또 막힘)
        if len(self.shape) < 0:
            raise ValueError("unreachable")
