from __future__ import annotations
import struct
from typing import Any, Dict

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved, OpFlags  # OpFlags 추가 임포트

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    P: int,
    G: int,
    M: int,
    V: int,
    bc1: int,
    bc2: int,
    outP: int,
    outM: int,
    outV: int,
    lr: float = 1e-3,
    beta1: float = 0.9,
    beta2: float = 0.999,
    eps: float = 1e-8,
    name: str = "adam_step",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """Adam Optimizer의 가중치 업데이트 연산을 IR에 기록합니다."""
    lr_f = float(lr)
    b1 = float(beta1)
    b2 = float(beta2)
    eps_f = float(eps)
    
    # ADAM Schema: [lr, beta1, beta2, eps] (f32 x 4)
    blob = struct.pack("<ffff", lr_f, b1, b2, eps_f)

    # ABI: 백엔드는 bc1, bc2를 rank0 스칼라 뷰로 기대함
    abi_hints = {"view_rank0_inputs": [4, 5]}
    if hints:
        abi_hints.update(hints)

    # 1. 정적 속성(Static Flags) 선언
    # AdamStep은 Optimizer이고, 상태를 변경하며, 그래프의 말단(Terminal)이다.
    static = OpFlags.IS_OPTIMIZER | OpFlags.HAS_STATE | OpFlags.TERMINAL
    
    # 기본 제약 조건 설정 및 Inplace 선호 비트 반영
    final_constraints = constraints or {"inplace_ok": True}
    if final_constraints.get("inplace_ok"):
        static |= OpFlags.INPLACE_PREF

    # 2. 통합 엔트리 호출 (static_flags 전달)
    return emit_resolved(
        b,
        kind="adam_step",
        name=name,
        inputs=[P, G, M, V, bc1, bc2],
        outputs=[outP, outM, outV],
        kind_id=ctx.AdamStep,
        attr_schema=ctx.SCHEMA_ADAM,
        attr_blob=blob,
        attrs={"lr": lr_f, "beta1": b1, "beta2": b2, "eps": eps_f},
        constraints=final_constraints,
        hints=abi_hints,
        static_flags=static, # 계산된 정적 비트 전달
    )

def emit_bwd(b: Builder, ctx: CudaEmitContext, fwd_node: Any, grad_y: int) -> Dict[int, int]:
    """Adam Step은 업데이트의 최종 단계이므로 역전파가 발생하지 않습니다."""
    return {}