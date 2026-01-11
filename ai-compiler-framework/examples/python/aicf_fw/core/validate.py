# aicf_fw/core/validate.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class IRValidationError(RuntimeError):
    """Raised when an IR graph violates required invariants."""


@dataclass(frozen=True)
class IRValidationReport:
    ok: bool
    warnings: list[str]


# ------------------------------------------------------------
# NEW: ops that are stateful / in-place (SSA exception)
#  - These ops may "write back" into an existing value id.
#  - Therefore:
#      * allow multiple producers for their outputs
#      * do not overwrite the first producer mapping (to avoid false use-before-define)
# ------------------------------------------------------------
_STATEFUL_INPLACE_OPS: set[str] = {"AdamStep", "StepInc", "BiasCorr"}


def validate_ir(ir: Any, *, ruleset: str = "train_v0") -> IRValidationReport:
    """
    Validate IR invariants used by the training step pipeline.

    Current ruleset ("train_v0"):
      - shape consistency for Linear/ReLU/MseGrad
      - topo + SSA single-producer + use-after-define
        (NOTE: stateful/in-place ops are exceptions)
      - Backward node inputs must reference existing values (if Backward exists)

    Returns:
      IRValidationReport(ok=True, warnings=[...])

    Raises:
      IRValidationError on hard failures.
    """
    warnings: list[str] = []

    if ruleset != "train_v0":
        raise ValueError(f"Unknown ruleset: {ruleset}")

    warnings.extend(_validate_ir_shape_consistency(ir))
    warnings.extend(_validate_ir_topo_ssa(ir))
    warnings.extend(_validate_ir_links_backward(ir))

    return IRValidationReport(ok=True, warnings=warnings)


# ------------------------------------------------------------
# Internal validators (ported from PR3 test)
# ------------------------------------------------------------

def _validate_ir_shape_consistency(ir: Any) -> list[str]:
    vals = ir.values

    def V(vid: Any):
        # PR3 used int(vid) indexing; support both int and str ids.
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
            if b is not None:
                if tuple(b.shape) != (OUT,):
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
            # v0: ignore unknown ops
            pass

    return []


def _validate_ir_topo_ssa(ir: Any) -> list[str]:
    """
    Invariants:
      - For normal ops: each value id must have a single producer.
      - For stateful/in-place ops: outputs may reuse existing value ids (SSA exception).
      - Use-after-define: if an input has a producer, that producer must appear before the consumer in topo order.
        (IMPORTANT: we keep the FIRST producer mapping; in-place ops must NOT overwrite it.)
    """
    produced_by: dict[Any, int] = {}

    # Pass 1: build producer map (FIRST producer wins)
    for node in ir.nodes:
        # In-place/stateful ops: do NOT register/overwrite producers for their outputs
        # (they "mutate" existing storage / ids)
        if getattr(node, "op", None) in _STATEFUL_INPLACE_OPS:
            continue

        for vid in node.outputs:
            if vid in produced_by:
                raise IRValidationError(
                    f"[ir][topo] value id={vid} has multiple producers: {produced_by[vid]} and {node.id}"
                )
            produced_by[vid] = node.id

    # Pass 2: use-after-define check (only when we know a producer)
    for node in ir.nodes:
        for vid in node.inputs:
            if vid in produced_by:
                prod = produced_by[vid]
                if prod >= node.id:
                    # node.id used as topo index in PR3; keep behavior
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
            warnings.append(
                f"[ir][links] Backward loss vid={loss_vid} is not produced by any node (might be okay in v0)"
            )
        if grad_vid is not None and grad_vid not in produced:
            warnings.append(
                f"[ir][links] Backward grad vid={grad_vid} is not produced by any node (might be okay in v0)"
            )

    return warnings
