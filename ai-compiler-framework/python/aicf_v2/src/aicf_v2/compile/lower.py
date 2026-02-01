from __future__ import annotations
from typing import List

from .types import LoweredOp
from ..backends.cuda.registry import CudaRegistry
from ..backends.cuda.attrs import pack_attrs
from ..builder import Builder


def lower_ir_cuda(b: Builder, registry: CudaRegistry) -> List[LoweredOp]:
    """
    IR op stream (b.ops) -> LoweredOp list.
    No rewrite/pass here.
    """
    lowered: List[LoweredOp] = []

    for op in b.ops:
        ks = registry.lookup(op.kind)

        # runtime_flags 결정은 plan 단계로 넘기는 게 정석.
        # 지금은 inplace 플래그는 false로 고정하고,
        # 실제 alias는 plan.alias에서 처리한다.
        attr_blob = pack_attrs(op.kind, op.attrs, runtime_flags={"inplace": False})

        lowered.append(
            LoweredOp(
                kind=op.kind,
                kind_id=ks.kind_id,
                attr_schema=ks.attr_schema,
                attr_blob=attr_blob,
                in_vids=list(op.inputs),
                out_vids=list(op.outputs),
                constraints=dict(getattr(op, "constraints", {}) or {}),
            )
        )

    return lowered
