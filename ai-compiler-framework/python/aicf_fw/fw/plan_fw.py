# aicf_fw/fw/plan_fw.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from aicf_fw.fw.emit_ctx import FrozenGraph, ValueDesc


@dataclass
class BindingPlanFW:
    """
    FW 전용 바인딩 플랜:
    value_id -> role/name/shape
    실행 시 inputs/params/statics/meta를 name으로 받아서,
    value_id -> torch.Tensor 로 매핑하는 lookup 테이블을 만든다.
    """
    name: str
    values: List[ValueDesc]
    # role별 name 리스트 (외부에서 어떤 텐서를 줘야 하는지)
    input_names: List[str]
    param_names: List[str]
    static_names: List[str]
    meta_names: List[str]


def build_binding_plan_fw(graph: FrozenGraph) -> BindingPlanFW:
    ins, ps, ss, ms = [], [], [], []
    for v in graph.values:
        if v.role == "input":
            ins.append(v.name)
        elif v.role == "param":
            ps.append(v.name)
        elif v.role == "static":
            ss.append(v.name)
        elif v.role == "meta":
            ms.append(v.name)
        else:
            # tmp는 외부 바인딩 필요 없음
            pass

    return BindingPlanFW(
        name=f"{graph.name}:binding_plan_fw",
        values=graph.values,
        input_names=ins,
        param_names=ps,
        static_names=ss,
        meta_names=ms,
    )


def bind_tensors_fw(
    plan: BindingPlanFW,
    *,
    inputs: Dict[str, Any],
    params: Dict[str, Any],
    statics: Dict[str, Any],
    meta: Dict[str, Any] | None = None,
) -> Dict[int, Any]:
    """
    name->tensor 딕셔너리들을 받아서 value_id->tensor로 변환.
    """
    name2tensor: Dict[str, Any] = {}
    name2tensor.update(params)
    name2tensor.update(statics)
    name2tensor.update(inputs)
    if meta:
        name2tensor.update(meta)

    vmap: Dict[int, Any] = {}
    for vid, v in enumerate(plan.values):
        if v.role in ("input", "param", "static", "meta"):
            if v.name not in name2tensor:
                raise KeyError(f"[bind] missing tensor for role={v.role} name={v.name}")
            vmap[vid] = name2tensor[v.name]
        else:
            # tmp는 실행 중에 생성됨
            vmap[vid] = None
    return vmap
