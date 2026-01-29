# aicf_fw/nn/relu.py
from __future__ import annotations

from aicf_fw.fw.module import Module
from aicf_fw.fw.emit_ctx import EmitCtx


class ReLU(Module):
    """
    forward: relu(x) -> y
    save_for_bwd=True:
      - create static saved buffer "<prefix>.saved"
      - emit "copy_saved" so relu_bwd can consume it
    """
    def __init__(self, save_for_bwd: bool = True):
        super().__init__()
        self.save_for_bwd = bool(save_for_bwd)

    def emit(self, ctx: EmitCtx, x_vid: int) -> int:
        pfx = self._prefix
        x_desc = ctx.values[x_vid]

        y_vid = ctx.new_value(
            name=f"{pfx}.out",
            shape=x_desc.shape,
            role="static",
        )

        ctx.emit_op(
            "relu",
            inputs=[x_vid],
            outputs=[y_vid],
            name=f"{pfx}.relu",
        )

        if self.save_for_bwd:
            saved_vid = ctx.static_vid(
                name=f"{pfx}.saved",
                shape=x_desc.shape,
            )
            ctx.emit_op(
                "copy_saved",
                inputs=[y_vid],
                outputs=[saved_vid],
                name=f"{pfx}.copy_saved",
            )

        return y_vid
