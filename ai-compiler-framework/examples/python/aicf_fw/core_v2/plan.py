from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import re

import torch

from .ir import IRGraph


# -----------------------------
# Binding roles (KEEP for printer.py)
# -----------------------------
ROLE_INPUT = "input"
ROLE_PARAM = "param"
ROLE_STATIC = "static"  # runtime allocate (temps/intermediates/outputs/states)


@dataclass
class BindSpec:
    vid: int
    role: str  # input|param|static
    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str


@dataclass
class BindingPlan:
    """
    inputs : 외부 주입 (x,t 등)
    params : 외부 주입 (모델 파라미터만: "0.W","0.b","2.W","2.b"...)
    statics: 런타임 allocate (intermediates, outputs, grads, optimizer states 등)
    """
    name: str
    specs: Dict[int, BindSpec] = field(default_factory=dict)

    inputs: List[int] = field(default_factory=list)
    params: List[int] = field(default_factory=list)
    statics: List[int] = field(default_factory=list)

    def role_of(self, vid: int) -> str:
        s = self.specs.get(int(vid))
        return s.role if s is not None else "unknown"


@dataclass
class PlanOptions:
    """
    Stage3/6 규칙(최종):
      - input_names: {"x","t"} 는 input
      - 모델 파라미터만 param: "숫자.(W|b)" 형태만
      - optimizer state는 무조건 static: "opt.*"
      - backward/grad는 무조건 static: "d_*", "grad*", "dY" 등
      - 나머지는 static
    """
    input_names: Set[str] = field(default_factory=lambda: {"x", "t"})

    # 모델 파라미터만 param 인정
    param_regex: str = r"^\d+\.(W|b)$"

    # 무조건 static 처리 prefix들
    static_prefixes: Tuple[str, ...] = (
        "d_", "grad", "dY",
        "opt.",           # ★ 핵심: optimizer state는 외부 주입 X, executor가 allocate해서 들고감
    )


def _compile_param_re(opts: PlanOptions) -> re.Pattern:
    return re.compile(opts.param_regex)


def _is_param_name(nm: str, opts: PlanOptions, param_re: re.Pattern) -> bool:
    # "0.W" / "2.b" 같은 모델 파라미터만 param
    return param_re.match(nm) is not None


def _is_forced_static(nm: str, opts: PlanOptions) -> bool:
    return any(str(nm).startswith(p) for p in opts.static_prefixes)


def build_binding_plan(ir: IRGraph, *, opts: Optional[PlanOptions] = None) -> BindingPlan:
    if opts is None:
        opts = PlanOptions()
    param_re = _compile_param_re(opts)

    plan = BindingPlan(name=f"{ir.name}:binding_plan")

    for vid, v in ir.values.items():
        vid = int(vid)
        nm = str(getattr(v, "name", f"v{vid}"))

        if nm in opts.input_names:
            role = ROLE_INPUT
        elif _is_forced_static(nm, opts):
            role = ROLE_STATIC
        elif _is_param_name(nm, opts, param_re):
            role = ROLE_PARAM
        else:
            role = ROLE_STATIC

        plan.specs[vid] = BindSpec(
            vid=vid,
            role=role,
            name=nm,
            shape=tuple(getattr(v, "shape", ())),
            dtype=str(getattr(v, "dtype", "torch.float32")),
            device=str(getattr(v, "device", "cuda")),
        )

    # stable lists
    plan.inputs.clear()
    plan.params.clear()
    plan.statics.clear()

    for vid in sorted(plan.specs.keys()):
        r = plan.specs[vid].role
        if r == ROLE_INPUT:
            plan.inputs.append(vid)
        elif r == ROLE_PARAM:
            plan.params.append(vid)
        else:
            plan.statics.append(vid)

    return plan


# -----------------------------
# allocate statics (with zero-init for opt states)
# -----------------------------
def _dtype_from_string(dtype_s: str) -> torch.dtype:
    ds = str(dtype_s)
    if "float16" in ds:
        return torch.float16
    if "bfloat16" in ds:
        return torch.bfloat16
    if "float32" in ds:
        return torch.float32
    if "float64" in ds:
        return torch.float64
    if "int64" in ds:
        return torch.int64
    if "int32" in ds:
        return torch.int32
    return torch.float32


def _should_zero_init_static(name: str) -> bool:
    """
    Optimizer state는 초기값이 의미 있음:
      - opt.step: 0
      - opt.m.*, opt.v.*: 0
    나머지 statics(intermediates/grads)는 empty로 충분.
    """
    if name.startswith("opt.step"):
        return True
    if name.startswith("opt.m.") or name.startswith("opt.v."):
        return True
    # 필요하면 bc1/bc2도 여기서 제어 가능:
    # if name.startswith("opt.bc1_inv") or name.startswith("opt.bc2_inv"):
    #     return False
    return False


def allocate_static_env(
    ir: IRGraph,
    plan: BindingPlan,
    *,
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """
    plan.statics에 해당하는 vid들을 allocate.
    - opt.* state는 zero-init
    - 나머지 static은 empty
    """
    env: Dict[int, torch.Tensor] = {}

    if device is None:
        d0 = None
        for vid in plan.statics:
            d0 = getattr(ir.values[int(vid)], "device", None)
            if d0:
                break
        try:
            device = torch.device(str(d0)) if d0 else torch.device("cuda")
        except Exception:
            device = torch.device("cuda")

    for vid in plan.statics:
        v = ir.values[int(vid)]
        nm = str(getattr(v, "name", f"v{vid}"))
        dt = _dtype_from_string(getattr(v, "dtype", "torch.float32"))
        shape = tuple(getattr(v, "shape", ()))

        if _should_zero_init_static(nm):
            env[int(vid)] = torch.zeros(shape, device=device, dtype=dt)
        else:
            env[int(vid)] = torch.empty(shape, device=device, dtype=dt)

    return env
