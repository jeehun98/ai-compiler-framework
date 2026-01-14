from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import torch

from .ir import IRGraph, IRValue


# -----------------------------
# Binding roles
# -----------------------------
ROLE_INPUT = "input"
ROLE_PARAM = "param"
ROLE_STATIC = "static"  # runtime allocate (temps/intermediates/outputs by default)


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
    BindingPlan = "누가 외부에서 바인딩되고, 누가 런타임에서 allocate 되는지"를 고정.
    - inputs: 외부 주입 (x,t 등)
    - params: 외부 주입 (W,b 등)
    - statics: runtime allocate (lin0_out, relu0_out, dY 등)
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
    Stage3 기본 규칙:
      - input_names: {"x","t"} 는 input
      - param name 규칙: "*.W" or "*.b" (또는 suffix로 W/b)
      - 나머지는 static
    """
    input_names: Set[str] = field(default_factory=lambda: {"x", "t"})
    param_suffixes: Tuple[str, ...] = (".W", ".b")
    param_exact_suffixes: Tuple[str, ...] = ("W", "b")  # fallback: "0.W"가 아니라 "W"인 경우


def _is_param_name(nm: str, opts: PlanOptions) -> bool:
    if any(nm.endswith(suf) for suf in opts.param_suffixes):
        return True
    if nm in opts.param_exact_suffixes:
        return True
    return False


def build_binding_plan(ir: IRGraph, *, opts: Optional[PlanOptions] = None) -> BindingPlan:
    if opts is None:
        opts = PlanOptions()

    plan = BindingPlan(name=f"{ir.name}:binding_plan")

    # Stage3에서는 "ir.values 전체"를 분류 대상으로 삼는다.
    # (Stage4에서 lowered-only subset 최적화 가능)
    for vid, v in ir.values.items():
        vid = int(vid)
        nm = str(v.name)

        if nm in opts.input_names:
            role = ROLE_INPUT
        elif _is_param_name(nm, opts):
            role = ROLE_PARAM
        else:
            role = ROLE_STATIC

        spec = BindSpec(
            vid=vid,
            role=role,
            name=nm,
            shape=tuple(v.shape),
            dtype=str(v.dtype),
            device=str(v.device),
        )
        plan.specs[vid] = spec

    # stable lists (sorted by vid)
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
# Optional helper: allocate statics
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


def allocate_static_env(
    ir: IRGraph,
    plan: BindingPlan,
    *,
    device: Optional[torch.device] = None,
) -> Dict[int, torch.Tensor]:
    """
    Stage3 helper:
      - plan.statics에 해당하는 vid들을 torch.empty로 allocate 해서 env dict로 반환.
      - inputs/params는 여기서 만들지 않는다(외부 주입 대상).
    """
    env: Dict[int, torch.Tensor] = {}

    # choose default device if not provided
    if device is None:
        # IRValue.device가 "cuda" 같은 문자열로 들어오는 케이스가 있으니 torch.device로 변환
        d0 = None
        for vid in plan.statics:
            d0 = ir.values[int(vid)].device
            if d0:
                break
        try:
            device = torch.device(str(d0)) if d0 else torch.device("cuda")
        except Exception:
            device = torch.device("cuda")

    for vid in plan.statics:
        v = ir.values[int(vid)]
        dt = _dtype_from_string(v.dtype)
        env[int(vid)] = torch.empty(tuple(v.shape), device=device, dtype=dt)

    return env
