from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...builder import Builder


def emit_resolved(
    b: Builder,
    *,
    kind: str,
    name: str,
    inputs: List[int],
    outputs: List[int],
    kind_id: int,
    attr_schema: int,
    attr_blob: bytes,
    attrs: Optional[Dict[str, Any]] = None,          # debug-friendly
    constraints: Optional[Dict[str, Any]] = None,
    hints: Optional[Dict[str, Any]] = None,
    saved: Optional[List[int]] = None,
) -> int:
    """
    Unified emitter entry:
      - calls Builder.emit()
      - writes backend-resolved caches onto Op: kind_id/attr_schema/attr_blob
      - preserves attrs/constraints/hints/saved for later passes/debug
    """
    op_index = b.emit(
        kind,
        inputs=list(inputs),
        outputs=list(outputs),
        name=str(name),
        attrs=dict(attrs or {}),
        constraints=dict(constraints or {}),
        hints=dict(hints or {}),
        saved=list(saved or []),
    )

    op = b.ops[op_index]

    # cache backend-resolved info on Op (even if Op dataclass doesn't declare them)
    setattr(op, "kind_id", int(kind_id))
    setattr(op, "attr_schema", int(attr_schema))
    setattr(op, "attr_blob", bytes(attr_blob))

    return op_index
