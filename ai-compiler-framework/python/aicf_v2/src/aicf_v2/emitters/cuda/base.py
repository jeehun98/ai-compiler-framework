from __future__ import annotations

from typing import Any, Dict, List, Optional

from ...builder import Builder


class OpFlags:
    """
    연산의 성격과 최적화 상태를 나타내는 비트마스크 정의.
    - 0~15 bits: Static Flags (Emitter가 설정하는 연산의 본질)
    - 16~31 bits: Derived Flags (Optimizer/Pass가 설정하는 그래프 맥락)
    """
    NONE = 0

    # --- Static Flags (Emitter-side) ---
    IS_GEMM_LIKE   = 1 << 0   # Gemm, Conv 등 무거운 행렬 연산
    IS_ELEMENTWISE = 1 << 1   # Add, Relu 등 1:1 매핑 연산
    IS_REDUCE      = 1 << 2   # Sum, Mean 등 차원 축소
    IS_OPTIMIZER   = 1 << 3   # Adam, SGD 등 업데이트 로직
    HAS_STATE      = 1 << 4   # 내부 버퍼를 직접 수정(Stateful)
    TERMINAL       = 1 << 5   # 그래프 종착점(Grad sink 등)
    INPLACE_PREF   = 1 << 6   # Inplace 실행을 권장
    
    # 신규(예시)
    IS_NORM        = 1 << 12     # normalization 계열 (BN/LN/RMSN 등)
    IS_BATCHNORM   = 1 << 13     # 필요하면 더 구체화

    # --- Derived Flags (Pass-side) ---
    SAFE_NODE      = 1 << 16  # Out-degree <= 1 등 퓨전 안전
    FUSION_BARRIER = 1 << 17  # 정책상 퓨전 차단
    DTYPE_F32      = 1 << 18  # 데이터 타입이 F32


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
    통합된 Emitter 엔트리:
    - Builder.emit()으로 노드 생성
    - 백엔드 실행에 필요한 로우레벨 정보(kind_id/schema/blob) 주입
    - attrs에 BWD가 참조할 FWD 정적 정보 보관(기존 의도 유지)
    - static_flags로 연산의 '본질'을 노드에 각인
    - derived_flags는 pass 단계에서 채우도록 0으로 초기화
    """
    op_index = b.emit(
        kind=kind,
        inputs=list(inputs),
        outputs=list(outputs),
        name=str(name),
        attrs=dict(attrs or {}),
        constraints=dict(constraints or {}),
        hints=hints,  # Builder 사양상 None 허용
        saved=list(saved or []),
    )

    op = b.ops[op_index]

    # C++ 백엔드 실행에 필요한 로우 레벨 정보 주입
    op.kind_id = int(kind_id)
    op.attr_schema = int(attr_schema)
    op.attr_blob = bytes(attr_blob)

    # 최적화/퓨전 판정을 위한 비트마스크 필드
    op.static_flags = int(static_flags)
    op.derived_flags = 0

    return op_index