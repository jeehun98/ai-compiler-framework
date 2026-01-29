# aicf_v2/fw/module.py
from __future__ import annotations
from typing import Dict, Optional

class Module:
    def __init__(self) -> None:
        self._prefix: str = ""
        self._children: Dict[str, "Module"] = {}

    def add_module(self, name: str, m: "Module") -> None:
        self._children[name] = m
        m._prefix = f"{self._prefix}.{name}" if self._prefix else name

    def emit(self, ctx, x_vid: int) -> int:
        raise NotImplementedError
