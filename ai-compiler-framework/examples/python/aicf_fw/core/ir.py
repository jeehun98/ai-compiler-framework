# aicf_fw/core/ir.py
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple
import json


@dataclass
class IRValue:
    id: int
    name: str
    shape: Tuple[int, ...]
    dtype: str
    device: str


@dataclass
class IRNode:
    id: int
    op: str
    inputs: List[int]          # IRValue ids
    outputs: List[int]         # IRValue ids
    attrs: Dict[str, Any]


class IRGraph:
    """
    NOTE:
      - IRValue.name는 실행/바인딩 ABI로도 쓰이므로 "유니크"해야 함.
      - 기존에는 W/b 같은 이름이 중복되어 IR-only 실행에서 bind 불가였음.
      - 여기서 name uniquify를 강제한다.
    """
    def __init__(self, name: str = "graph"):
        self.name = name
        self._next_val_id = 0
        self._next_node_id = 0
        self.values: Dict[int, IRValue] = {}
        self.nodes: List[IRNode] = []

        # name uniquifier: "W" -> "W", "W#1", "W#2" ...
        self._name_count: Dict[str, int] = {}

    def _uniq_name(self, name: str) -> str:
        base = str(name)
        c = self._name_count.get(base, 0)
        self._name_count[base] = c + 1
        return base if c == 0 else f"{base}#{c}"

    def new_value(self, *, name: str, shape: Tuple[int, ...], dtype: str, device: str) -> IRValue:
        vid = self._next_val_id
        self._next_val_id += 1

        nm = self._uniq_name(name)

        v = IRValue(
            id=vid,
            name=nm,
            shape=tuple(shape),
            dtype=str(dtype),
            device=str(device),
        )
        self.values[vid] = v
        return v

    def emit(
        self,
        *,
        op: str,
        inputs: List[IRValue],
        outputs: List[IRValue],
        attrs: Optional[Dict[str, Any]] = None,
    ) -> IRNode:
        nid = self._next_node_id
        self._next_node_id += 1
        node = IRNode(
            id=nid,
            op=str(op),
            inputs=[v.id for v in inputs],
            outputs=[v.id for v in outputs],
            attrs=dict(attrs or {}),
        )
        self.nodes.append(node)
        return node

    def to_dict(self) -> Dict[str, Any]:
        return {
            "graph": self.name,
            "values": {str(k): asdict(v) for k, v in self.values.items()},
            "nodes": [asdict(n) for n in self.nodes],
        }

    def dump_json(self, *, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=False)


# ---- validation (moved from core/validate.py) ----

class IRValidationError(RuntimeError):
    """Raised when an IR graph violates required invariants."""


@dataclass(frozen=True)
class IRValidationReport:
    ok: bool
    warnings: list[str]


# NEW: stateful / in-place ops (SSA exception)
# - allow multiple producers for their outputs
# - do not overwrite the first producer mapping
_STATEFUL_INPLACE_OPS: set[str] = {"AdamStep", "StepInc", "BiasCorr"}


def validate_ir(ir: Any, *, ruleset: str = "train_v0") -> IRValidationReport:
    """
    Validate IR invariants used by the training step pipeline.

    ruleset "train_v0":
      - shape consistency for Linear/ReLU/MseGrad
      - topo + SSA single-producer + use-after-define
        (NOTE: stateful/in-place ops are exceptions)
      - Backward node inputs must reference existing values (if Backward exists)

    Returns: IRValidationReport(ok=True, warnings=[...])
    Raises: IRValidationError on hard failures.
    """
    warnings: list[str] = []

    if ruleset != "train_v0":
        raise ValueError(f"Unknown ruleset: {ruleset}")

    warnings.extend(_validate_ir_shape_consistency(ir))
    warnings.extend(_validate_ir_topo_ssa(ir))
    warnings.extend(_validate_ir_links_backward(ir))

    return IRValidationReport(ok=True, warnings=warnings)


def _validate_ir_shape_consistency(ir: Any) -> list[str]:
    vals = ir.values

    def V(vid: Any):
        try:
            return vals[int(vid)]
        except Exception:
            return vals[vid]

    for node in ir.nodes:
        op = node.op
        ins = node.inputs
        outs = node.outputs

        if op == "Linear":
            if len(ins) not in (2, 3):
                raise IRValidationError(f"[ir][shape] Linear expects 2 or 3 inputs, got {len(ins)}")
            if len(outs) != 1:
                raise IRValidationError(f"[ir][shape] Linear expects 1 output, got {len(outs)}")

            x = V(ins[0])
            W = V(ins[1])
            b = V(ins[2]) if len(ins) == 3 else None
            y = V(outs[0])

            if len(x.shape) != 2 or len(W.shape) != 2:
                raise IRValidationError(f"[ir][shape] Linear requires 2D x/W: x={x.shape} W={W.shape}")
            B, IN = x.shape
            OUT, IN2 = W.shape
            if IN != IN2:
                raise IRValidationError(f"[ir][shape] Linear K mismatch: x={x.shape} W={W.shape}")
            if b is not None and tuple(b.shape) != (OUT,):
                raise IRValidationError(
                    f"[ir][shape] Linear bias shape mismatch: b={b.shape} expected={(OUT,)}"
                )
            if tuple(y.shape) != (B, OUT):
                raise IRValidationError(
                    f"[ir][shape] Linear out shape mismatch: y={y.shape} expected={(B, OUT)}"
                )

        elif op == "ReLU":
            if len(ins) != 1 or len(outs) != 1:
                raise IRValidationError(
                    f"[ir][shape] ReLU expects 1 in/1 out, got in={len(ins)} out={len(outs)}"
                )
            x = V(ins[0])
            y = V(outs[0])
            if tuple(x.shape) != tuple(y.shape):
                raise IRValidationError(f"[ir][shape] ReLU shape mismatch: x={x.shape} y={y.shape}")

        elif op == "MseGrad":
            if len(ins) != 2 or len(outs) != 1:
                raise IRValidationError(
                    f"[ir][shape] MseGrad expects 2 in/1 out, got in={len(ins)} out={len(outs)}"
                )
            p = V(ins[0])
            t = V(ins[1])
            o = V(outs[0])
            if tuple(p.shape) != tuple(t.shape):
                raise IRValidationError(
                    f"[ir][shape] MseGrad pred/target mismatch: pred={p.shape} tgt={t.shape}"
                )
            if tuple(o.shape) != tuple(p.shape):
                raise IRValidationError(
                    f"[ir][shape] MseGrad out shape mismatch: out={o.shape} expected={p.shape}"
                )

        else:
            pass

    return []


def _validate_ir_topo_ssa(ir: Any) -> list[str]:
    """
    Invariants:
      - normal ops: single producer per value id
      - stateful/in-place ops: outputs may reuse existing value ids
      - use-after-define: if input has producer, producer must appear before consumer
        (FIRST producer wins; in-place ops must not overwrite it)
    """
    produced_by: dict[Any, int] = {}

    # Pass 1: build producer map (FIRST producer wins)
    for node in ir.nodes:
        if getattr(node, "op", None) in _STATEFUL_INPLACE_OPS:
            continue

        for vid in node.outputs:
            if vid in produced_by:
                raise IRValidationError(
                    f"[ir][topo] value id={vid} has multiple producers: {produced_by[vid]} and {node.id}"
                )
            produced_by[vid] = node.id

    # Pass 2: use-after-define
    for node in ir.nodes:
        for vid in node.inputs:
            if vid in produced_by:
                prod = produced_by[vid]
                if prod >= node.id:
                    raise IRValidationError(
                        f"[ir][topo] node {node.id}({node.op}) uses value id={vid} "
                        f"before its producer ({prod}, '{ir.nodes[prod].op}')"
                    )

    return []


def _validate_ir_links_backward(ir: Any) -> list[str]:
    warnings: list[str] = []
    vals = ir.values

    produced = set()
    for n in ir.nodes:
        for o in n.outputs:
            produced.add(o)

    backs = [n for n in ir.nodes if n.op == "Backward"]
    if not backs:
        warnings.append("[ir] no Backward node found (ok if v0 omits it)")
        return warnings

    for bn in backs:
        if len(bn.inputs) not in (1, 2):
            raise IRValidationError(f"[ir][links] Backward expects 1 or 2 inputs, got {len(bn.inputs)}")
        loss_vid = bn.inputs[0]
        grad_vid = bn.inputs[1] if len(bn.inputs) == 2 else None

        if loss_vid not in vals:
            raise IRValidationError(f"[ir][links] Backward loss value missing: {loss_vid}")
        if grad_vid is not None and grad_vid not in vals:
            raise IRValidationError(f"[ir][links] Backward grad value missing: {grad_vid}")

        if loss_vid not in produced:
            warnings.append(f"[ir][links] Backward loss vid={loss_vid} is not produced by any node (v0이면 ok)")
        if grad_vid is not None and grad_vid not in produced:
            warnings.append(f"[ir][links] Backward grad vid={grad_vid} is not produced by any node (v0이면 ok)")

    return warnings
