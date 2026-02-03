from __future__ import annotations

from .types import CompiledProgram
from .plan import make_exec_plan_cuda
from ..model import Model
from ..backends.cuda.registry import CudaRegistry
from .passes.pipeline import optimize_ir


def compile_cuda(m: Model, registry: CudaRegistry) -> CompiledProgram:
    """
    compile(Model) -> CompiledProgram(plan)
    """
    b0 = m.b
    b1 = optimize_ir(b0)

    plan = make_exec_plan_cuda(b1, registry=registry)  # <- 네가 lower 제거 방향이면 이렇게
    
    return CompiledProgram(plan=plan)
