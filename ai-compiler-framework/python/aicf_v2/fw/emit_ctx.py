# aicf_v2/fw/emit_ctx.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

ValueId = int
BufferId = int

@dataclass(frozen=True)
class TensorSpec:
    shape: Tuple[int, ...]
    dtype: str               # "f16", "f32"
    device: str = "cuda"

@dataclass(frozen=True)
class IRNode:
    op: str
    inputs: Tuple[ValueId, ...]
    outputs: Tuple[ValueId, ...]
    name: str
    attrs: Dict[str, Any]

@dataclass
class Program:
    # IR
    value_specs: List[TensorSpec] = field(default_factory=list)
    value_names: List[str] = field(default_factory=list)
    nodes: List[IRNode] = field(default_factory=list)

    # Lowered
    buffer_specs: List[TensorSpec] = field(default_factory=list)      # buffer_id -> spec
    value_to_buffer: Dict[ValueId, BufferId] = field(default_factory=dict)
    lowered_ops: List[Dict[str, Any]] = field(default_factory=list)   # minimal dict form


@dataclass
class EmitPolicy:
    force_kernel_id: Optional[str] = None  # 디버그/강제 선택

@dataclass
class EmitCtx:
    """
    v2 목표:
    - 레이어는 ctx.new_value / ctx.param_vid / ctx.emit_op 만 사용
    - ctx.emit_op이 IR + lowered 동시 생성
    """
    B: int
    dtype: str = "f16"
    device: str = "cuda"
    policy: EmitPolicy = field(default_factory=EmitPolicy)
    prog: Program = field(default_factory=Program)

    _param_name_to_vid: Dict[str, ValueId] = field(default_factory=dict)

    # -----------------------------
    # values
    # -----------------------------
    def new_value(self, name: str, shape: Tuple[int, ...], role: str = "static") -> ValueId:
        # role은 지금은 metadata일 뿐 (나중에 planner에서 활용)
        spec = TensorSpec(shape=tuple(shape), dtype=self.dtype, device=self.device)
        vid = len(self.prog.value_specs)
        self.prog.value_specs.append(spec)
        self.prog.value_names.append(name)

        # value는 기본적으로 동일 spec의 새 buffer를 가진다 (심볼)
        bid = len(self.prog.buffer_specs)
        self.prog.buffer_specs.append(spec)
        self.prog.value_to_buffer[vid] = bid
        return vid

    def param_vid(self, name: str, shape: Tuple[int, ...]) -> ValueId:
        # 같은 파라미터 name이면 같은 vid/buffer 재사용
        if name in self._param_name_to_vid:
            return self._param_name_to_vid[name]

        vid = self.new_value(name=name, shape=shape, role="param")
        self._param_name_to_vid[name] = vid
        return vid

    def _vid_to_bid(self, vid: ValueId) -> BufferId:
        try:
            return self.prog.value_to_buffer[vid]
        except KeyError as e:
            raise KeyError(f"ValueId {vid} has no buffer mapping") from e

    # -----------------------------
    # emit op: IR + lowered 동시 생성
    # -----------------------------
    def emit_op(
        self,
        op: str,
        inputs: List[ValueId],
        outputs: List[ValueId],
        name: str,
        **attrs: Any,
    ) -> None:
        # 1) IR
        self.prog.nodes.append(
            IRNode(
                op=op,
                inputs=tuple(inputs),
                outputs=tuple(outputs),
                name=name,
                attrs=dict(attrs),
            )
        )

        # 2) Lowered (초기엔 op별 if로 단순하게)
        in_bids = [self._vid_to_bid(v) for v in inputs]
        out_bids = [self._vid_to_bid(v) for v in outputs]

        if op == "gemm":
            kernel = self.policy.force_kernel_id or "gemm_simple_v1"
            launch = {"grid": (1, 1, 1), "block": (128, 1, 1), "smem": 0}
            # args: [A, B, C]
            args = [in_bids[0], in_bids[1], out_bids[0]]
            meta = {"transA": bool(attrs.get("transA", False)), "transB": bool(attrs.get("transB", False))}
        elif op == "bias_add":
            kernel = self.policy.force_kernel_id or "bias_add_axis_last_v1"
            launch = {"grid": (1, 1, 1), "block": (256, 1, 1), "smem": 0}
            # args: [X, B, Y]
            # inplace=True면 out bid == in bid를 허용 (여기선 그냥 out_bids[0] 쓰면 됨)
            args = [in_bids[0], in_bids[1], out_bids[0]]
            meta = {
                "inplace": bool(attrs.get("inplace", False)),
                "broadcast_axis": int(attrs.get("broadcast_axis", -1)),
            }
        else:
            raise NotImplementedError(f"emit_op lower not implemented for op={op}")

        self.prog.lowered_ops.append(
            {
                "name": name,
                "op": op,
                "kernel": kernel,
                "launch": launch,
                "args": tuple(args),
                "meta": meta,
            }
        )
