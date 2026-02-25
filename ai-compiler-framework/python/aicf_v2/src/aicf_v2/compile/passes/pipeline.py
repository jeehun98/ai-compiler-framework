from __future__ import annotations
from ..types import ExecPlan
from ...builder import Builder

# 분리한 fusion 패스를 가져옵니다.
from .fusion import RichMatrixOptimizer, GraphRewriter

def optimize_ir(b: Builder) -> Builder:
    """
    AICF_V2 Graph Optimization Pipeline
    """
    # 1. 정보가 풍부한 인접 행렬 생성
    opt = RichMatrixOptimizer(b)
    opt.encode()
    
    # 2. 특정 패턴(Gemm+Bias+ReLU) 탐색
    targets = opt.find_fused_gemm_bias_relu()
    
    # 3. 발견된 타겟이 있다면 그래프 변조 적용
    if targets:
        rewriter = GraphRewriter(b)
        rewriter.apply_fusion(targets)
        
        # 4. (선택) 무효화된 노드들을 완전히 제거 (DCE: Dead Code Elimination)
        # b.ops = [op for op in b.ops if op.kind != "nop"]
        
    return b

def optimize_plan(plan: ExecPlan) -> ExecPlan:
    """
    실행 계획 최적화 (스케줄링, 메모리 등)
    """
    return plan