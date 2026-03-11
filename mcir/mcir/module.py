from __future__ import annotations

from dataclasses import dataclass, field

from .regions import Region


@dataclass
class MCModule:
    regions: list[Region] = field(default_factory=list)