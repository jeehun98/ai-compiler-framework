from __future__ import annotations
from typing import Any, Dict, List, Optional
from ...builder import Builder

class OpFlags:
    """
    연산의 성격(Semantics)과 그래프 내의 역할(Traits)을 비트로 정의.
    - 0~15 bits: Static Flags (연산 고유의 속성)
    - 16~31 bits: Derived Flags (그래프 위상 및 런타임 정보)
    """
    NONE = 0

    # --- [Static Flags: 성격 (Semantics)] ---
    IS_GEMM_LIKE     = 1 << 0   # Gemm, Conv 등 대량 연산 (Fusion Root 후보)
    IS_ELEMENTWISE   = 1 << 1   # 1:1 매핑 연산
    IS_REDUCE        = 1 << 2   # 차원 축소 연산
    IS_OPTIMIZER     = 1 << 3   # 상태 업데이트 연산
    IS_NORM          = 1 << 4   # Normalization 계열 (BN, LN 등)
    IS_ACTIVATION    = 1 << 5   # ReLU, Sigmoid 등 활성화 함수

    # --- [Static Flags: 특성 (Traits/Roles)] ---
    # 패턴 매칭 시 if문 대신 비트로 역할을 즉시 판별
    HAS_BIAS         = 1 << 8   # Bias 입력을 가지는 연산 (예: bias_add)
    HAS_STATE        = 1 << 9   # 내부 버퍼 수정 (Stateful)
    INPLACE_PREF     = 1 << 10  # Inplace 실행 권장
    TERMINAL         = 1 << 11  # 그래프 종착점 (Grad sink)

    # --- [Derived Flags: 맥락 (Context/Pass)] ---
    SAFE_NODE        = 1 << 16  # Out-degree <= 1 (Fusion 안전)
    FUSION_BARRIER   = 1 << 17  # 정책상 퓨전 차단 지점
    DTYPE_F32        = 1 << 18  # FP32 데이터 타입
    DTYPE_F16        = 1 << 19  # FP16/BF16 데이터 타입
    
    # --- [Helper Mask: 패턴 쿼리용] ---
    # 예: Gemm Epilogue 후보 (Gemm 성격이면서 데이터 타입이 일치)
    QUERY_GEMM_ROOT = IS_GEMM_LIKE | DTYPE_F32

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
    static_flags: int = OpFlags.NONE,
) -> int:
    """
    모든 연산의 공통 진입점.
    static_flags를 통해 '지문(Fingerprint)'을 각인하여 
    Pass 단계에서 정밀 검증(문자열 비교) 횟수를 최소화함.
    """
    op_index = b.emit(
        kind=kind,
        inputs=list(inputs),
        outputs=list(outputs),
        name=str(name),
        attrs=dict(attrs or {}),
        constraints=dict(constraints or {}),
        hints=hints,
        saved=list(saved or []),
    )

    op = b.ops[op_index]
    op.kind_id = int(kind_id)
    op.attr_schema = int(attr_schema)
    op.attr_blob = bytes(attr_blob)

    # Emitter에서 정의한 연산의 본질 주입
    op.static_flags = int(static_flags)
    # Optimizer가 분석할 공간 확보
    op.derived_flags = 0

    return op_index