from __future__ import annotations
from typing import Any

from ..types import ExecPlan
from ...builder import Builder

# 고도화된 RichMatrixOptimizer, GraphRewriter와 패턴 정의를 가져옵니다.
from .fusion import RichMatrixOptimizer, GraphRewriter, FUSION_PATTERNS

def optimize_ir(b: Builder, ctx: Any) -> Builder:
    """
    AICF_V2 Graph Optimization Pipeline
    
    Args:
        b: 최적화 대상인 그래프가 담긴 Builder 객체
        ctx: 백엔드 전용 정보를 담고 있는 CudaEmitContext 객체 (fused emitter 호출에 필요)
        
    1. RichMatrixOptimizer를 통해 그래프의 위상(CSR)과 노드별 성격(Bitmask)을 분석합니다.
    2. FUSION_PATTERNS에 선언된 비트 시퀀스들을 기반으로 퓨전 대상 후보를 탐색합니다.
    3. GraphRewriter를 사용하여 발견된 패턴들을 최적화된 Fused Emitter 노드로 치환합니다.
    """
    
    # 1. 인접 행렬 및 노드 비트마스크 생성 (지문 각인 단계)
    opt = RichMatrixOptimizer(b)
    opt.encode()
    
    # 2. 그래프 수정을 담당할 Rewriter 준비 (ctx 주입)
    # GraphRewriter 내부에서 self.ctx로 보관하여 gemm_epilogue.emit 시 사용합니다.
    rewriter = GraphRewriter(b, ctx)

    # 3. 선언적 패턴 매칭 실행
    found_any = False
    for pattern_name in FUSION_PATTERNS.keys():
        # 비트 시퀀스 매칭을 통해 퓨전 대상 인덱스 그룹들을 찾음
        matches = opt.find_matches(pattern_name)
        
        if matches:
            # 발견된 타겟이 있다면 해당 패턴 명세에 따라 그래프 변조 적용
            rewriter.apply_fusion(pattern_name, matches)
            found_any = True
            
    # 4. (선택적) 추가 패스 실행 지점 (예: Dead Code Elimination)
    if found_any:
        # 퓨전 이후 'nop'으로 변한 노드들을 실제로 제거하거나 메모리를 정리할 수 있습니다.
        pass
        
    return b

def optimize_plan(plan: ExecPlan) -> ExecPlan:
    """
    실행 계획 최적화 (Memory Planning, Kernel Scheduling 등)
    """
    return plan