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
    attrs: Optional[Dict[str, Any]] = None,
    constraints: Optional[Dict[str, Any]] = None,
    hints: Optional[Dict[str, Any]] = None,
    saved: Optional[List[int]] = None,
) -> int:
    """
    통합된 Emitter 엔트리:
    - Builder.emit()을 호출하여 그래프에 노드를 추가합니다.
    - 백엔드에서 해석된 ID와 바이너리 블롭을 Op 객체에 직접 주입(setattr)합니다.
    - [중요] attrs를 통해 BWD가 참조할 FWD의 정적 정보를 보관합니다.
    """
    op_index = b.emit(
        kind=kind,
        inputs=list(inputs),
        outputs=list(outputs),
        name=str(name),
        attrs=dict(attrs or {}),
        constraints=dict(constraints or {}),
        hints=hints, # hints는 None 허용 (Builder 사양에 따름)
        saved=list(saved or []),
    )

    op = b.ops[op_index]

    # C++ 백엔드 실행에 필요한 로우 레벨 정보 주입
    op.kind_id = int(kind_id)
    op.attr_schema = int(attr_schema)
    op.attr_blob = bytes(attr_blob)

    return op_index