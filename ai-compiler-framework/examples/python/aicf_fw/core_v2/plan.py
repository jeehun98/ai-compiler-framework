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
    - params: 외부 주입 (W,b 등)  <-- 진짜 파라미터만!
    - statics: runtime allocate (intermediates, outputs, grads 등)
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
    Stage3 기본 규칙(수정):
      - input_names: {"x","t"} 는 input
      - param name 규칙: "0.W", "0.b", "2.W", "2.b" 같은 layer 파라미터만 param
      - "d_*", "grad*", "dY" 등은 절대 param이 아니고 static
      - 나머지는 static
    """
    input_names: Set[str] = field(default_factory=lambda: {"x", "t"})

    # "0.W" / "2.b" 형태만 param 인정 (이게 핵심)
    param_regex: str = r"^\d+\.(W|b)$"

    # backward/grad 네이밍 prefix는 무조건 static으로
    static_prefixes: Tuple[str, ...] = ("d_", "grad", "dY")


_PARAM_RE = re.compile(PlanOptions().param_regex)


def _is_param_name(nm: str, opts: PlanOptions) -> bool:
    """
    IMPORTANT:
      - 이전처럼 '.W'로 끝나면 다 param 처리하면 "d_2.W" 같은 게 param으로 오염됨.
      - 이제는 '숫자. W/b' 형태만 param.
    """
    if _PARAM_RE.match(nm) is not None:
        return True
    return False


def _is_forced_static(nm: str, opts: PlanOptions) -> bool:
    return any(str(nm).startswith(p) for p in opts.static_prefixes)


def build_binding_plan(ir: IRGraph, *, opts: Optional[PlanOptions] = None) -> BindingPlan:
    if opts is None:
        opts = PlanOptions()

    plan = BindingPlan(name=f"{ir.name}:binding_plan")

    # Stage3에서는 ir.values 전체를 분류 대상으로 삼는다.
    for vid, v in ir.values.items():
        vid = int(vid)
        nm = str(getattr(v, "name", f"v{vid}"))

        if nm in opts.input_names:
            role = ROLE_INPUT
        elif _is_forced_static(nm, opts):
            role = ROLE_STATIC
        elif _is_param_name(nm, opts):
            role = ROLE_PARAM
        else:
            role = ROLE_STATIC

        spec = BindSpec(
            vid=vid,
            role=role,
            name=nm,
            shape=tuple(getattr(v, "shape", ())),
            dtype=str(getattr(v, "dtype", "torch.float32")),
            device=str(getattr(v, "device", "cuda")),
        )
        plan.specs[vid] = spec

    # stable lists (sorted by vid)
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
    plan.statics에 해당하는 vid들을 torch.empty로 allocate.
    inputs/params는 외부 주입.
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
        dt = _dtype_from_string(getattr(v, "dtype", "torch.float32"))
        env[int(vid)] = torch.empty(tuple(getattr(v, "shape", ())), device=device, dtype=dt)

    return env
