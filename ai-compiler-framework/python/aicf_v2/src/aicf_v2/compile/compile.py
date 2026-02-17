# python/aicf_v2/src/aicf_v2/compile/compile.py

from __future__ import annotations
from typing import TYPE_CHECKING, Optional

from .types import CompiledProgram
from .plan import make_exec_plan_cuda
from .passes.pipeline import optimize_ir

# 타입 체크 시에만 Model을 인식하도록 설정하여 순환 참조 방지
if TYPE_CHECKING:
    from ..model import Model
    from ..backends.cuda.registry import CudaRegistry


def compile_cuda(m: 'Model', registry: Optional['CudaRegistry'] = None) -> CompiledProgram:
    """
    Model 객체를 받아 실행 가능한 CompiledProgram(ExecPlan 포함)으로 변환합니다.

    1. optimize_ir: IR 수준의 그래프 최적화 (Passes)
    2. plan: 메모리 슬롯 할당 및 In-place(Alias) 결정
    """
    # 순환 참조 방지를 위해 함수 내부에서 필요한 경우에만 참조하거나 
    # m.b (Builder) 객체에만 접근합니다.
    
    # 1. IR 최적화 단계 (현재는 identity pass)
    b0 = m.b
    b1 = optimize_ir(b0)

    # 2. 실행 계획 수립 (In-place Alias 결정)
    # 이제 plan 단계에서 하드웨어 registry가 직접적으로 필요하지 않도록 설계되었습니다.
    plan = make_exec_plan_cuda(b1)
    
    return CompiledProgram(plan=plan)