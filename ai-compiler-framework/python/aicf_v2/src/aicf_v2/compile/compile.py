from __future__ import annotations

from .types import CompiledProgram
from .plan import make_exec_plan_cuda
from .passes.pipeline import optimize_ir
from ..model import Model
from ..backends.cuda.registry import CudaRegistry


def compile_cuda(m: Model, registry: CudaRegistry) -> CompiledProgram:
    """
    compile(Model) -> CompiledProgram(plan)

    - optimize_ir: Builder-level rewrite hook (currently identity)
    - plan: alias/inplace decisions (no lower stage)
    """
    b0 = m.b
    b1 = optimize_ir(b0)

    plan = make_exec_plan_cuda(b1)  # ✅ registry 인자 제거
    return CompiledProgram(plan=plan)
