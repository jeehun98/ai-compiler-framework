from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class LoweredOp:
    """
    Backend-ready call spec for _C.op_call(...)
    """
    kind: str
    kind_id: int
    attr_schema: int
    attr_blob: bytes
    in_vids: List[int]
    out_vids: List[int]
    constraints: Dict[str, Any]
    hints: Dict[str, Any] = field(default_factory=dict)  # ✅ NEW


@dataclass
class ExecPlan:
    """
    Execution plan = lowered ops + runtime decisions (alias/inplace/etc.)
    """
    lowered: List[LoweredOp]
    alias: Dict[int, int]  # out_vid -> in_vid (slot alias)


@dataclass
class CompiledProgram:
    plan: ExecPlan
