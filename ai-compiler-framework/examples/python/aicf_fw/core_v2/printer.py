from __future__ import annotations

from typing import Any, Dict

from .ir import IRGraph


def dump_ir(ir: IRGraph) -> str:
    lines = []
    lines.append(f"=== IRGraph: {ir.name} ===")
    lines.append(f"values: {len(ir.values)}, nodes: {len(ir.nodes)}")
    lines.append("")

    # values
    lines.append("[values]")
    for vid in sorted(ir.values.keys()):
        v = ir.values[vid]
        lines.append(
            f"  v{v.id:03d}  {v.name:12s} shape={tuple(v.shape)} dtype={v.dtype} device={v.device}"
        )

    lines.append("")
    lines.append("[nodes]")
    for i, n in enumerate(ir.nodes):
        ins = ", ".join([f"v{vid:03d}" for vid in n.inputs])
        outs = ", ".join([f"v{vid:03d}" for vid in n.outputs])
        attrs = _fmt_attrs(n.attrs)
        lines.append(f"  #{i:03d} {n.op:8s}  ({ins}) -> ({outs}){attrs}")

    return "\n".join(lines)


def _fmt_attrs(attrs: Dict[str, Any]) -> str:
    if not attrs:
        return ""
    # 짧게
    items = []
    for k, v in attrs.items():
        if isinstance(v, float):
            items.append(f"{k}={v:.6g}")
        else:
            items.append(f"{k}={v}")
    return "  {" + ", ".join(items) + "}"
