# aicf_fw/nn/linear.py (추가/수정 아이디어)
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

from aicf_fw.core_v2.backend_ops import BackendOp


@dataclass
class Linear:
    in_features: int
    out_features: int
    bias: bool = True
    device: str | None = None
    dtype: object | None = None
    name: str = ""

    # 기존에 이미 파라미터 관리 로직이 있을 텐데(0.W / 0.b 등),
    # 여기서는 "컴파일 템플릿 생성"에 필요한 param name만 사용한다고 가정.
    # (실제 파라미터 텐서는 fw.module 쪽이 보유)
    def param_names(self, prefix: str) -> dict:
        # Sequential에서 prefix="0", "2" 같은 걸 넘겨줄 수 있게
        names = {"W": f"{prefix}.W"}
        if self.bias:
            names["b"] = f"{prefix}.b"
        return names

    def lower_template(self, ctx, x_vid: int, prefix: str) -> tuple[int, List[BackendOp]]:
        """
        Linear forward 템플릿을 BackendOp로 생성.
        반환: (y_vid, ops)
        """
        p = self.param_names(prefix)
        W_vid = ctx.param_vid(p["W"])          # ctx가 param name -> value id 제공
        b_vid = ctx.param_vid(p.get("b", "")) if self.bias else None

        # 1) GEMM
        # AICF에서 linear이 y = x @ W^T + b 형태였으니 transB=True로 둠.
        y_vid = ctx.new_vid(name=f"{prefix}.linear_out")  # output value id 생성(또는 ctx.emit이 생성)
        ops: List[BackendOp] = []
        ops.append(
            BackendOp(
                op_kind="gemm",
                inputs=[x_vid, W_vid],
                outputs=[y_vid],
                attrs={"transA": False, "transB": True},
                name=f"{prefix}.gemm",
            )
        )

        # 2) BIAS_ADD (inplace로 y에 누적)
        if self.bias and b_vid is not None:
            ops.append(
                BackendOp(
                    op_kind="bias_add",
                    inputs=[y_vid, b_vid],
                    outputs=[y_vid],  # inplace
                    attrs={"inplace": True, "broadcast_axis": -1},
                    name=f"{prefix}.bias_add",
                )
            )

        return y_vid, ops
