# examples/python/aicf_fw/core_v2/opt_rules.py
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .ir import IRGraph


def _op(it: Dict[str, Any]) -> str:
    return str(it.get("op", "")).strip().lower()


def _ins(it: Dict[str, Any]) -> List[int]:
    return [int(x) for x in it.get("inputs", [])]


def _outs(it: Dict[str, Any]) -> List[int]:
    return [int(y) for y in it.get("outputs", [])]


def _attrs(it: Dict[str, Any]) -> Dict[str, Any]:
    return dict(it.get("attrs", {}) or {})


def _is_inplace_unary(it: Dict[str, Any]) -> bool:
    ins = _ins(it)
    outs = _outs(it)
    return (len(ins) >= 1 and len(outs) == 1 and ins[0] == outs[0])


def _clone_item(it: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "op": _op(it),
        "inputs": _ins(it),
        "outputs": _outs(it),
        "attrs": _attrs(it),
        **{k: v for k, v in it.items() if k not in ("op", "inputs", "outputs", "attrs")},
    }


def _try_fuse_gemm_bias_relu(lowered: List[Dict[str, Any]], i: int) -> Tuple[bool, int, Dict[str, Any]]:
    """
    Match:
      gemm(x, W) -> y
      bias_add(y, b) -> y   (inplace)
      relu(y) -> z

    Replace with:
      gemm_epilogue(x, W, b) -> z   attrs: {transA/transB..., epilogue="bias_relu"}
    """
    if i + 2 >= len(lowered):
        return (False, 0, {})

    a = lowered[i]
    b = lowered[i + 1]
    c = lowered[i + 2]

    if _op(a) != "gemm" or _op(b) != "bias_add" or _op(c) != "relu":
        return (False, 0, {})

    a_ins = _ins(a)
    a_outs = _outs(a)
    if len(a_ins) != 2 or len(a_outs) != 1:
        return (False, 0, {})

    y = a_outs[0]

    if not _is_inplace_unary(b):
        return (False, 0, {})
    b_ins = _ins(b)
    if len(b_ins) != 2 or b_ins[0] != y:
        return (False, 0, {})
    bias = b_ins[1]

    c_ins = _ins(c)
    c_outs = _outs(c)
    if len(c_ins) != 1 or len(c_outs) != 1 or c_ins[0] != y:
        return (False, 0, {})
    z = c_outs[0]

    fused = {
        "op": "gemm_epilogue",
        "inputs": [a_ins[0], a_ins[1], bias],  # x, W, b
        "outputs": [z],
        "attrs": {
            **_attrs(a),
            "epilogue": "bias_relu",
        },
    }
    return (True, 3, fused)


def apply_opt_rules(ir: IRGraph, lowered: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Stage Opt: rewrite / fuse / normalize.

    Current rules:
      - fuse gemm + bias_add + relu -> gemm_epilogue (bias+relu)

    NOTE:
      - We only fuse patterns that match the kernels you actually registered:
        OpKind::GemmEpilogue has gemm_bias_relu_* only (no bias-only kernel registered yet).
    """
    out: List[Dict[str, Any]] = []
    i = 0
    lowered = [_clone_item(it) for it in lowered]

    while i < len(lowered):
        ok, consumed, fused = _try_fuse_gemm_bias_relu(lowered, i)
        if ok:
            out.append(fused)
            i += consumed
            continue

        out.append(lowered[i])
        i += 1

    return out
