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
    attr_schema: int = 0,
    attr_blob: bytes = b"",
    attrs: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    hints: Optional[Dict[str, Any]] = None,
    saved: Optional[List[int]] = None,
) -> int:
    """
    Emit an op with ABI-resolved fields filled at emit-time.

    - kind_id: must match C++ OpKind value
    - attr_schema/attr_blob: must match backend ABI
    - attrs: keep semantic/debug attrs (optional)
    """
    if kind_id is None:
        raise ValueError(f"emit_resolved: kind_id must be provided (kind={kind}, name={name})")
    if attr_blob is None:
        raise ValueError(f"emit_resolved: attr_blob must be bytes (kind={kind}, name={name})")

    return b.emit(
        kind,
        inputs=inputs,
        outputs=outputs,
        name=name,
        attrs=attrs,
        constraints=constraints,
        hints=hints,
        saved=saved,
        kind_id=int(kind_id),
        attr_schema=int(attr_schema),
        attr_blob=attr_blob,
    )
