from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .values import MCValue


@dataclass
class MCNode:
    name: str
    op: str
    inputs: list[MCValue] = field(default_factory=list)
    outputs: list[MCValue] = field(default_factory=list)
    attrs: dict[str, Any] = field(default_factory=dict)