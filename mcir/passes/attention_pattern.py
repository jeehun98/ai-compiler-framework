from __future__ import annotations

from mcir.builder import MCIRBuilder
from mcir.module import MCModule


def run_attention_pattern_pass(semantic_graph) -> MCModule:
    b = MCIRBuilder()
    module = b.module()

    ops = semantic_graph.ops
    if len(ops) < 3:
        return module

    op_types = [op.op_type for op in ops]

    has_basic_pattern = (
        len(op_types) >= 3
        and op_types[0] == "matmul"
        and "softmax" in op_types
        and op_types[-1] == "matmul"
    )

    if not has_basic_pattern:
        return module

    q = semantic_graph.tensors["Q"]
    k = semantic_graph.tensors["K"]
    v = semantic_graph.tensors["V"]
    o = semantic_graph.tensors["O"]

    qv = b.value("Q", q.shape, q.dtype, residency="global")
    kv = b.value("K", k.shape, k.dtype, residency="global")
    vv = b.value("V", v.shape, v.dtype, residency="global")
    ov = b.value("O", o.shape, o.dtype, residency="global")

    region = b.execution_region("attention_region")
    region.inputs.extend([qv, kv, vv])
    region.outputs.append(ov)

    region.attrs["pattern"] = "scaled_dot_product_attention"
    region.attrs["has_mask"] = "mask" in op_types
    region.attrs["has_softmax"] = "softmax" in op_types

    module.regions.append(region)
    return module