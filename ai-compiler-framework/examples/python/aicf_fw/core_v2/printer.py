from __future__ import annotations

from typing import Any, Dict, List, Optional

from .ir import IRGraph
from .plan import BindingPlan, ROLE_INPUT, ROLE_PARAM, ROLE_STATIC


def _fmt_attrs(attrs: Dict[str, Any]) -> str:
    if not attrs:
        return ""
    items = []
    for k, v in attrs.items():
        if isinstance(v, float):
            items.append(f"{k}={v:.6g}")
        else:
            items.append(f"{k}={v}")
    return "  {" + ", ".join(items) + "}"


def dump_ir(ir: IRGraph) -> str:
    lines: List[str] = []
    lines.append(f"=== IRGraph: {ir.name} ===")
    lines.append(f"values: {len(ir.values)}, nodes: {len(ir.nodes)}")
    lines.append("")

    lines.append("[values]")
    for vid in sorted(ir.values.keys()):
        v = ir.values[vid]
        lines.append(
            f"  v{int(v.id):03d}  {str(v.name):12s} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
        )

    lines.append("")
    lines.append("[nodes]")
    for i, n in enumerate(ir.nodes):
        ins = ", ".join([f"v{int(vid):03d}" for vid in n.inputs])
        outs = ", ".join([f"v{int(vid):03d}" for vid in n.outputs])
        attrs = _fmt_attrs(n.attrs or {})
        # op 폭은 stage 커지면 8로 부족할 수 있어서 10으로 살짝 늘림
        lines.append(f"  #{i:03d} {str(n.op):10s} ({ins}) -> ({outs}){attrs}")

    return "\n".join(lines)


def dump_lowered(
    lowered: List[Dict[str, Any]],
    *,
    title: str = "LoweredOps",
    name: Optional[str] = None,
) -> str:
    """
    - title: 기본 헤더 이름 (기존 호환)
    - name: stage 이름 같은 라벨 (신규). 주어지면 'LoweredOps(name)' 형태로 표시.
    """
    hdr = f"{title}({name})" if name else title

    lines: List[str] = []
    lines.append(f"=== {hdr} ===")
    lines.append(f"ops: {len(lowered)}")
    lines.append("")

    for i, it in enumerate(lowered):
        op = str(it.get("op"))
        ins = [int(x) for x in it.get("inputs", [])]
        outs = [int(y) for y in it.get("outputs", [])]
        attrs = dict(it.get("attrs", {}) or {})

        ins_s = ", ".join([f"v{v:03d}" for v in ins])
        outs_s = ", ".join([f"v{v:03d}" for v in outs])
        attrs_s = _fmt_attrs(attrs)

        lines.append(f"  #{i:03d} {op:10s} ({ins_s}) -> ({outs_s}){attrs_s}")

    return "\n".join(lines)


def dump_plan(
    plan: BindingPlan,
    *,
    title: str = "BindingPlan",
    name: Optional[str] = None,
) -> str:
    """
    - title: 기본 헤더 이름 (기존 호환)
    - name : stage 라벨 (예: v2_stage6_train1)
    """
    hdr = f"{title}({name})" if name else title

    lines: List[str] = []
    lines.append(f"=== {hdr} ===")
    lines.append(f"name: {plan.name}")
    lines.append(
        f"inputs: {len(plan.inputs)}, params: {len(plan.params)}, statics: {len(plan.statics)}"
    )
    lines.append("")

    def _sec(h: str, vids: List[int]):
        lines.append(f"[{h}]")
        for vid in vids:
            s = plan.specs[int(vid)]
            lines.append(
                f"  v{s.vid:03d}  {s.name:12s} role={s.role:6s} "
                f"shape={tuple(s.shape)} dtype={s.dtype} device={s.device}"
            )
        lines.append("")

    _sec("inputs", plan.inputs)
    _sec("params", plan.params)
    _sec("statics", plan.statics)

    return "\n".join(lines)
