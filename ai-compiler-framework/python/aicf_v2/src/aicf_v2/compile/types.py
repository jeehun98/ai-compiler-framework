from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List

from ..graph import Op


@dataclass
class ExecPlan:
    """
    Execution plan = op stream + runtime decisions (alias/inplace/etc.)

    ops:
      - builder가 만든 Op 리스트를 그대로 들고 간다.
      - op.kind_id/attr_schema/attr_blob/hints 는 emitter가 채운다.

    alias:
      - out_vid -> in_vid (slot alias)
    """
    ops: List[Op]
    alias: Dict[int, int]


@dataclass
class CompiledProgram:
    plan: ExecPlan
