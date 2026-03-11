from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class Tensor:
    name: str
    shape: tuple[int, ...]
    dtype: str = "fp16"


@dataclass
class Op:
    name: str
    op_type: str
    inputs: List[str]
    outputs: List[str]
    attrs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticGraph:
    tensors: Dict[str, Tensor] = field(default_factory=dict)
    ops: List[Op] = field(default_factory=list)

    def add_tensor(self, tensor: Tensor) -> None:
        self.tensors[tensor.name] = tensor

    def add_op(self, op: Op) -> None:
        self.ops.append(op)


def build_attention_semantic_graph() -> SemanticGraph:
    g = SemanticGraph()

    g.add_tensor(Tensor("Q", (1, 8, 128, 64), "fp16"))
    g.add_tensor(Tensor("K", (1, 8, 128, 64), "fp16"))
    g.add_tensor(Tensor("V", (1, 8, 128, 64), "fp16"))

    g.add_tensor(Tensor("scores", (1, 8, 128, 128), "fp16"))
    g.add_tensor(Tensor("masked_scores", (1, 8, 128, 128), "fp16"))
    g.add_tensor(Tensor("probs", (1, 8, 128, 128), "fp16"))
    g.add_tensor(Tensor("O", (1, 8, 128, 64), "fp16"))

    g.add_op(Op(
        name="score_matmul",
        op_type="matmul",
        inputs=["Q", "K"],
        outputs=["scores"],
        attrs={"transpose_b": True},
    ))

    g.add_op(Op(
        name="mask",
        op_type="mask",
        inputs=["scores"],
        outputs=["masked_scores"],
        attrs={"causal": True},
    ))

    g.add_op(Op(
        name="softmax",
        op_type="softmax",
        inputs=["masked_scores"],
        outputs=["probs"],
        attrs={"axis": -1},
    ))

    g.add_op(Op(
        name="value_matmul",
        op_type="matmul",
        inputs=["probs", "V"],
        outputs=["O"],
        attrs={"transpose_b": False},
    ))

    return g