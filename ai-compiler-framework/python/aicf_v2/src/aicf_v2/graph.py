from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Literal

from .tensor_spec import TensorSpec

ValueRole = Literal["input", "param", "state", "tmp", "output"]


@dataclass
class Value:
    vid: int
    name: str
    spec: TensorSpec
    producer_op: Optional[int] = None
    users: List[int] = field(default_factory=list)

    # role for CUDA Graph-friendly policies
    role: ValueRole = "tmp"


@dataclass
class Op:
    kind: str
    name: str
    inputs: List[int]
    outputs: List[int]

    attrs: Dict[str, Any] = field(default_factory=dict)

    # 레이어가 최종 결정을 박지 않고, "가능/선호"만 남기는 공간
    constraints: Dict[str, Any] = field(default_factory=dict)
    hints: Dict[str, Any] = field(default_factory=dict)

    # training용: "무엇을 저장해야 하는지"만 선언 (실제 alloc/copy/alias는 compile/plan에서)
    saved: List[int] = field(default_factory=list)

    # -------------------------
    # NEW: optional ABI/registry cache (emit-time pre-resolve / pre-pack)
    # -------------------------
    # - kind_id: C++ enum OpKind value (stable ABI contract if you guarantee it)
    # - attr_schema: fourcc(...) or 0
    # - attr_blob: pack_attrs(kind, attrs, ...) result
    #
    # These are caches: optimizer passes that change kind/attrs MUST invalidate them.
    kind_id: Optional[int] = None
    attr_schema: Optional[int] = None
    attr_blob: Optional[bytes] = None


def invalidate_op_cache(op: Op) -> None:
    """Invalidate ABI/registry caches when op.kind/op.attrs change."""
    op.kind_id = None
    op.attr_schema = None
    op.attr_blob = None
