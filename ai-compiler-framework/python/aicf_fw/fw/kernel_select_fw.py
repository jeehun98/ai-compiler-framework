# aicf_fw/fw/kernel_select_fw.py
from __future__ import annotations

from typing import List, Dict, Any

from aicf_fw.fw.emit_ctx import FrozenGraph


def _is_f16(dt) -> bool:
    s = str(dt)
    return ("float16" in s) or ("torch.float16" in s) or ("Half" in s)


def _dtype_of_value(graph: FrozenGraph, vid: int):
    return graph.values[int(vid)].dtype


def fill_kernel_ids_fw(graph: FrozenGraph) -> None:
    """
    아주 단순한 StageB:
    - dtype만 보고 kid를 채움
    - vec2/half2 같은 세부 조건은 나중에 추가
    """
    for op in graph.ops:
        if op.kernel_id is not None:
            continue

        k = op.op_kind
        in0dt = _dtype_of_value(graph, op.inputs[0]) if op.inputs else None
        out0dt = _dtype_of_value(graph, op.outputs[0]) if op.outputs else None
        f16 = _is_f16(in0dt) or _is_f16(out0dt)

        if k == "gemm":
            op.kernel_id = "gemm_f16_tc_wmma_out_f16_v0" if f16 else "gemm_f32_naive_v0"
        elif k == "gemm_epilogue":
            op.kernel_id = "gemm_bias_relu_f16_tc_wmma_out_f16_v0" if f16 else "gemm_bias_relu_f32_naive_v0"
        elif k == "bias_add":
            op.kernel_id = "bias_add_f16_v0" if f16 else "bias_add_f32_v0"
        elif k == "relu":
            op.kernel_id = "relu_f16_v0" if f16 else "relu_f32_v0"
        elif k == "relu_bwd":
            op.kernel_id = "relu_bwd_f16_v0" if f16 else "relu_bwd_f32_v0"
        elif k == "mse_grad":
            op.kernel_id = "mse_grad_f16_v0" if f16 else "mse_grad_f32_v0"
        elif k == "reduce_sum":
            # keep_lastdim 계열 가정
            op.kernel_id = "reduce_sum_keep_lastdim_f16_v0" if f16 else "reduce_sum_keep_lastdim_f32_v0"
        elif k == "sgd_step":
            op.kernel_id = "sgd_step_f16_v0" if f16 else "sgd_step_f32_v0"
        elif k == "adam_step":
            op.kernel_id = "adam_step_f32_v0"
        elif k in ("copy", "copy_saved", "copy_aux"):
            op.kernel_id = "copy_f16_v0" if f16 else "copy_f32_v0"
        elif k == "grad_zero":
            op.kernel_id = "grad_zero_v0"
        elif k == "step_inc":
            op.kernel_id = "step_inc_v0"
        elif k in ("bias_corr", "biascorr"):
            op.kernel_id = "bias_corr_v0"
        else:
            raise RuntimeError(f"[StageB FW] unknown op_kind={k}")
