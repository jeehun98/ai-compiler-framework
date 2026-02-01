from __future__ import annotations

from .types import CompiledProgram
from .lower import lower_ir_cuda
from .plan import make_exec_plan_cuda
from .passes.pipeline import optimize_ir, optimize_plan
from ..model import Model
from ..backends.cuda.registry import CudaRegistry


def compile_cuda(m: Model, registry: CudaRegistry) -> CompiledProgram:
    """
    compile(Model) -> CompiledProgram(plan)
    현재: optimize_ir는 identity.
    """
    b0 = m.b
    b1 = optimize_ir(b0)

    lowered = lower_ir_cuda(b1, registry)
    plan = make_exec_plan_cuda(b1, lowered)
    plan = optimize_plan(plan)

    return CompiledProgram(plan=plan)
