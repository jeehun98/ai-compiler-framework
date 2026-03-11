from __future__ import annotations

from .module import MCModule
from .nodes import MCNode
from .regions import Region


def _indent(depth: int) -> str:
    return "  " * depth


def _format_attrs(attrs: dict) -> str:
    if not attrs:
        return ""
    items = ", ".join(f"{k}={v}" for k, v in attrs.items())
    return f" attrs: {items}"


def _dump_node(node: MCNode, depth: int, out: list[str]) -> None:
    out.append(f"{_indent(depth)}Node({node.name}, op={node.op}){_format_attrs(node.attrs)}")
    if node.inputs:
        ins = ", ".join(v.short() for v in node.inputs)
        out.append(f"{_indent(depth + 1)}inputs: {ins}")
    if node.outputs:
        outs = ", ".join(v.short() for v in node.outputs)
        out.append(f"{_indent(depth + 1)}outputs: {outs}")


def _dump_region(region: Region, depth: int, out: list[str]) -> None:
    out.append(f"{_indent(depth)}{region.__class__.__name__}({region.name}){_format_attrs(region.attrs)}")

    if region.inputs:
        ins = ", ".join(v.short() for v in region.inputs)
        out.append(f"{_indent(depth + 1)}inputs: {ins}")

    if region.outputs:
        outs = ", ".join(v.short() for v in region.outputs)
        out.append(f"{_indent(depth + 1)}outputs: {outs}")

    for node in region.nodes:
        _dump_node(node, depth + 1, out)

    for sub in region.subregions:
        _dump_region(sub, depth + 1, out)


def dump_module(module: MCModule) -> str:
    out: list[str] = ["MCModule"]
    for region in module.regions:
        _dump_region(region, 1, out)
    return "\n".join(out)