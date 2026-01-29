# aicf_v2/nn/linear.py
from __future__ import annotations

from aicf_v2.fw.module import Module
from aicf_v2.fw.emit_ctx import EmitCtx


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.bias = bool(bias)

    def emit(self, ctx: EmitCtx, x_vid: int) -> int:
        pfx = self._prefix or "linear"

        W_vid = ctx.param_vid(f"{pfx}.W", shape=(self.out_features, self.in_features))

        y_vid = ctx.new_value(
            name=f"{pfx}.out",
            shape=(ctx.B, self.out_features),
            role="static",
        )

        ctx.emit_op(
            "gemm",
            inputs=[x_vid, W_vid],
            outputs=[y_vid],
            name=f"{pfx}.gemm",
            transA=False,
            transB=True,
        )

        if self.bias:
            b_vid = ctx.param_vid(f"{pfx}.b", shape=(self.out_features,))
            ctx.emit_op(
                "bias_add",
                inputs=[y_vid, b_vid],
                outputs=[y_vid],
                name=f"{pfx}.bias_add",
                inplace=True,
                broadcast_axis=-1,
            )

        return y_vid
