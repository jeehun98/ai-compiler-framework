from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MCValue:
    name: str
    shape: tuple[int, ...]
    dtype: str
    residency: str = "global"
    producer: Optional[str] = None
    consumers: list[str] = field(default_factory=list)

    def short(self) -> str:
        shape_str = ",".join(str(x) for x in self.shape)
        return f"{self.name}[{shape_str}]:{self.dtype}@{self.residency}"