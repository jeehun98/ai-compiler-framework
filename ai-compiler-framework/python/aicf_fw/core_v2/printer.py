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


def dump_lowered(lowered, name: str = "lowered") -> str:
    """
    LoweredOps pretty printer.
    - StageA: op_kind/inputs/outputs/attrs
    - StageB: +kernel_id(+kind name)까지 함께 표시
    """
    lines = []
    lines.append(f"=== LoweredOps({name}) ===")

    # lowered가 list[dict] 또는 list[LoweredOp] 둘 다 커버(최대한 관대하게)
    ops = lowered if isinstance(lowered, (list, tuple)) else getattr(lowered, "ops", lowered)
    n_ops = len(ops) if hasattr(ops, "__len__") else 0
    lines.append(f"ops: {n_ops}\n")

    def _fmt_io(vs):
        if vs is None:
            return ""
        # list/tuple of value ids
        if isinstance(vs, (list, tuple)):
            return ", ".join(str(x) for x in vs)
        return str(vs)

    def _fmt_attrs(attrs):
        if not attrs:
            return ""
        # attrs가 dict면 key 정렬해서 안정적으로 출력
        if isinstance(attrs, dict):
            items = []
            for k in sorted(attrs.keys()):
                items.append(f"{k}={attrs[k]}")
            return "{" + ", ".join(items) + "}"
        # 그 외는 repr
        return "{" + repr(attrs) + "}"

    for i, op in enumerate(ops):
        # dict 형태
        if isinstance(op, dict):
            kind = op.get("kind") or op.get("op") or op.get("op_kind") or "unknown"
            ins = op.get("inputs") or op.get("in") or op.get("args") or []
            outs = op.get("outputs") or op.get("out") or op.get("rets") or []
            attrs = op.get("attrs") or {}
            kid = op.get("kernel_id", None)

        # 객체 형태
        else:
            kind = getattr(op, "kind", None) or getattr(op, "op", None) or getattr(op, "op_kind", None) or "unknown"
            ins = getattr(op, "inputs", None) or getattr(op, "in_", None) or []
            outs = getattr(op, "outputs", None) or getattr(op, "out", None) or []
            attrs = getattr(op, "attrs", None) or {}
            kid = getattr(op, "kernel_id", None)

        io = f"({_fmt_io(ins)}) -> ({_fmt_io(outs)})"
        a = _fmt_attrs(attrs)
        if a:
            base = f"  #{i:03d} {kind:<10} {io}  {a}"
        else:
            base = f"  #{i:03d} {kind:<10} {io}"

        # ✅ StageB 핵심: kernel_id를 같이 찍기
        if kid is not None:
            base += f"  kid={kid}"

        lines.append(base)

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
