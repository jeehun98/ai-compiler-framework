from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
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
      - optimizer state는 기본 static: "opt.*"
      - backward/grad는 기본 static: "d_*", "grad*", "dY" 등
      - 나머지는 static

    NOTE:
      - host-managed meta(opt.bc1_inv/opt.bc2_inv 등)를 외부에서 주입하고 싶으면
        PlanOptions.static_prefixes에 걸리더라도 executor의 bind 로직이 role mismatch를
        완화(fallback)해서 실제 텐서를 env[vid]에 꽂을 수 있다.
      - “외부 주입 필수 static”을 강제하고 싶으면 아래 ExecPlanOptions에서 제어한다.
    """
    input_names: Set[str] = field(default_factory=lambda: {"x", "t"})

    # 모델 파라미터만 param 인정
    param_regex: str = r"^\d+\.(W|b)$"

    # 무조건 static 처리 prefix들
    static_prefixes: Tuple[str, ...] = (
        "d_", "grad", "dY",
        "opt.",  # optimizer/state/meta는 기본 static
    )


def _compile_param_re(opts: PlanOptions) -> re.Pattern:
    return re.compile(opts.param_regex)


def _is_param_name(nm: str, opts: PlanOptions, param_re: re.Pattern) -> bool:
    # "0.W" / "2.b" 같은 모델 파라미터만 param
    return param_re.match(nm) is not None


def _is_forced_static(nm: str, opts: PlanOptions) -> bool:
    return any(str(nm).startswith(p) for p in opts.static_prefixes)


def build_binding_plan(ir: IRGraph, *, opts: Optional[PlanOptions] = None) -> BindingPlan:
    """
    BindingPlan은 “메모리 역할”만 정의한다.
    (결정/커널 선택은 ExecPlan 단계에서 별도로 한다.)
    """
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


# ============================================================
# ✅ ExecPlan: 결정(커널/전략)을 lowered에 “의미적으로 박제”하는 단계
# ============================================================

@dataclass
class ExecPlanOptions:
    """
    이 옵션은 'binding'이 아니라 '결정 박제'에 관한 것.

    목표:
      - lowered op마다 kernel_id를 채워서
        executor가 runtime dispatch(op_call) 없이 launch_by_id로 실행할 수 있게 한다.

    현재 구현은 '스켈레톤 + 최소 규약' 제공용:
      - gemm: transA/transB를 kernel_id에 반영
      - reduce_sum: axis/keepdim을 반영(없으면 default)
      - adam_step: schema_id=ADAM + attrs(lr,beta1,beta2,eps)는 별도 bytes로 가고,
        kernel_id는 "adam_step.default" 같은 형태로 고정(추후 variant화 가능)

    NOTE:
      - 실제 registry의 kernel_id 명명 규칙이 정해지면,
        아래 _select_kernel_id_* 함수들에서 그 규칙으로 바꾸면 된다.
    """
    # arch 문자열을 plan에서 직접 박제하고 싶으면 설정
    # 예: "sm_75", "sm_80"
    arch: Optional[str] = None

    # kernel_id가 없던 lowered에 자동으로 채워넣을지 여부
    # True면 apply_kernel_decisions가 강제로 채움
    # False면 이미 있는 kernel_id만 유지
    fill_missing_kernel_id: bool = True

    # 디버깅용: kernel_id에 상세 정보(예: shape 등)를 넣고 싶으면 True
    verbose_kernel_id: bool = False


def apply_kernel_decisions(
    lowered: List[Dict[str, Any]],
    *,
    opts: Optional[ExecPlanOptions] = None,
) -> List[Dict[str, Any]]:
    """
    lowered(list[dict])를 입력으로 받아 각 op에 kernel_id를 채운 복사본을 반환한다.

    lowered item 최소 형식(기존 유지):
      {
        "op": "gemm",
        "inputs": [v0, v1],
        "outputs": [v2],
        "attrs": {...}
      }

    결정 박제 후 형식(추가 필드):
      {
        ...,
        "kernel_id": "gemm.sm_80.ta0.tb1"  # 예시
      }
    """
    if opts is None:
        opts = ExecPlanOptions()

    out: List[Dict[str, Any]] = []
    for it in lowered:
        it2 = dict(it)
        op = str(it2.get("op", "")).strip().lower()

        # 이미 kernel_id가 있으면 그대로 유지
        if it2.get("kernel_id", None) is not None:
            out.append(it2)
            continue

        if not opts.fill_missing_kernel_id:
            out.append(it2)
            continue

        attrs = dict(it2.get("attrs", {}) or {})
        kid = _select_kernel_id(op, attrs, opts)
        it2["kernel_id"] = kid
        out.append(it2)

    return out


def _select_kernel_id(op: str, attrs: Dict[str, Any], opts: ExecPlanOptions) -> str:
    if op == "gemm":
        return _select_kernel_id_gemm(attrs, opts)
    if op == "reduce_sum":
        return _select_kernel_id_reduce_sum(attrs, opts)
    if op == "adam_step":
        return _select_kernel_id_adam(attrs, opts)

    # 그 외 op들은 일단 default로 박제 (추후 variant화)
    # bias_add / relu / relu_bwd / mse_grad / copy 등
    return _kid_base(op, opts) + ".default"


def _kid_base(op: str, opts: ExecPlanOptions) -> str:
    # 예: "gemm.sm_80" / "relu.sm_75" / "adam_step"
    if opts.arch:
        return f"{op}.{opts.arch}"
    return op


def _select_kernel_id_gemm(attrs: Dict[str, Any], opts: ExecPlanOptions) -> str:
    ta = 1 if bool(attrs.get("transA", False)) else 0
    tb = 1 if bool(attrs.get("transB", False)) else 0
    base = _kid_base("gemm", opts)
    # 최소로 trans flags만 박제
    kid = f"{base}.ta{ta}.tb{tb}"
    return kid


def _select_kernel_id_reduce_sum(attrs: Dict[str, Any], opts: ExecPlanOptions) -> str:
    axis = attrs.get("axis", None)
    keepdim = bool(attrs.get("keepdim", False))
    base = _kid_base("reduce_sum", opts)

    if axis is None:
        kid = f"{base}.axis?.k{1 if keepdim else 0}"
    else:
        kid = f"{base}.axis{int(axis)}.k{1 if keepdim else 0}"
    return kid


def _select_kernel_id_adam(attrs: Dict[str, Any], opts: ExecPlanOptions) -> str:
    # adam은 보통 알고리즘/타일링 variant가 따로 존재할 수 있지만,
    # 지금 단계에서는 “결정이 plan에 박제된다”는 사실이 중요하므로 default로 고정.
    base = _kid_base("adam_step", opts)
    return f"{base}.default"


def _dtype_str(v) -> str:
    dt = getattr(v, "dtype", "torch.float32")
    return str(dt)


def _is_f16(dtype_s: str) -> bool:
    s = str(dtype_s)
    return ("float16" in s) or ("torch.float16" in s) or ("Half" in s)


def _last_dim(ir: IRGraph, vid: int) -> int:
    v = ir.values[int(vid)]
    shape = tuple(getattr(v, "shape", ()))
    if not shape:
        return 1
    return int(shape[-1])

def _numel(ir: IRGraph, vid: int) -> int:
    v = ir.values[int(vid)]
    shape = tuple(getattr(v, "shape", ()))
    if not shape:
        return 1
    n = 1
    for d in shape:
        n *= int(d)
    return int(n)


def _pick_vec2_upgrade(
    op: str,
    base_kid: Optional[str],
    in_dtype: str,
    out_dtype: str,
    lastdim: int,
    *,
    in_numel: int,
    out_numel: int,
) -> Optional[str]:
    """
    Stage B: vec2/half2 같은 "형상 기반 결정"을 planner에서 박제한다.

    - add/bias_add/relu/relu_bwd: lastdim 짝수면 vec2
    - sgd_step: numel 짝수면 half2  (lastdim 기준 말고 flat 기준)
    """
    op = str(op).strip().lower()
    f16 = _is_f16(in_dtype) or _is_f16(out_dtype)

    # f16 아니면 업그레이드 없음
    if not f16:
        return base_kid

    # add/bias_add/relu/relu_bwd 는 lastdim 기반 vec2
    even_ld = (int(lastdim) % 2 == 0)
    if op == "add" and even_ld:
        return "add_f16_vec2_v0"
    if op == "bias_add" and even_ld:
        return "bias_add_f16_vec2_v0"
    if op == "relu" and even_ld:
        return "relu_f16_vec2_v0"
    if op == "relu_bwd" and even_ld:
        return "relu_bwd_f16_vec2_v0"

    # ✅ sgd_step은 flat numel 기반 half2
    if op == "sgd_step":
        even_numel = (int(out_numel) % 2 == 0)  # out_vid == param vid라 out 기준이 자연스러움
        if even_numel:
            return "sgd_step_f16_half2_v0"
        return base_kid

    return base_kid


def apply_kernel_decisions_stageB(ir: IRGraph, lowered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []

    for it in lowered:
        it = dict(it)
        op = str(it.get("op", "")).strip().lower()
        in_vids = [int(x) for x in it.get("inputs", [])]
        out_vids = [int(y) for y in it.get("outputs", [])]

        in_dtype = _dtype_str(ir.values[in_vids[0]]) if in_vids else ""
        out_dtype = _dtype_str(ir.values[out_vids[0]]) if out_vids else ""

        ld = _last_dim(ir, out_vids[0]) if out_vids else (_last_dim(ir, in_vids[0]) if in_vids else 1)

        in_numel = _numel(ir, in_vids[0]) if in_vids else 1
        out_numel = _numel(ir, out_vids[0]) if out_vids else 1

        base_kid = it.get("kernel_id", None)

        new_kid = _pick_vec2_upgrade(
            op, base_kid, in_dtype, out_dtype, ld,
            in_numel=in_numel, out_numel=out_numel,
        )

        if new_kid is not None:
            it["kernel_id"] = str(new_kid)

        out.append(it)

    return out
