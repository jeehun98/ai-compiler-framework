from __future__ import annotations
from typing import List, Optional

from .types import LoweredOp
from ..builder import Builder
from ..backends.cuda.registry import CudaRegistry


def lower_ir_cuda(
    b: Builder,
    registry: Optional[CudaRegistry] = None,
    *,
    verify: bool = False,
) -> List[LoweredOp]:
    """
    IR op stream (b.ops) -> LoweredOp list. (Emitter-first)

    Contract:
      - emitters must fill Op.kind_id / Op.attr_schema / Op.attr_blob.
      - lower is a thin serialization step (no registry lookup, no pack).

    verify mode:
      - if verify=True, registry must be provided
      - validate kind_id/attr_schema against registry mapping (debug safety net)
    """
    if verify and registry is None:
        raise ValueError("lower_ir_cuda: verify=True requires registry")

    lowered: List[LoweredOp] = []

    for op in b.ops:
        # -------------------------
        # Require emitter-filled caches
        # -------------------------
        if getattr(op, "kind_id", None) is None:
            raise ValueError(f"[lower_ir_cuda] missing op.kind_id (kind='{op.kind}', name='{op.name}')")
        if getattr(op, "attr_schema", None) is None:
            # schema를 완전히 없앨 계획이면 이 체크 삭제 가능
            raise ValueError(f"[lower_ir_cuda] missing op.attr_schema (kind='{op.kind}', name='{op.name}')")
        if getattr(op, "attr_blob", None) is None:
            raise ValueError(f"[lower_ir_cuda] missing op.attr_blob (kind='{op.kind}', name='{op.name}')")

        kind_id = int(op.kind_id)
        attr_schema = int(op.attr_schema)
        attr_blob = op.attr_blob

        # -------------------------
        # Optional verification (debug)
        # -------------------------
        if verify:
            ks = registry.lookup(op.kind)  # type: ignore[union-attr]
            if ks.kind_id != kind_id:
                raise ValueError(
                    f"[lower_ir_cuda] kind_id mismatch: kind='{op.kind}', op={kind_id}, reg={ks.kind_id}"
                )
            if ks.attr_schema != attr_schema:
                raise ValueError(
                    f"[lower_ir_cuda] attr_schema mismatch: kind='{op.kind}', op={attr_schema}, reg={ks.attr_schema}"
                )

        lowered.append(
            LoweredOp(
                kind=op.kind,  # 디버그용. 필요 없으면 LoweredOp에서 제거 가능
                kind_id=kind_id,
                attr_schema=attr_schema,
                attr_blob=attr_blob,
                in_vids=list(op.inputs),
                out_vids=list(op.outputs),
                constraints=dict(getattr(op, "constraints", {}) or {}),
            )
        )

    return lowered
