from __future__ import annotations
import struct
from typing import Dict, Any

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved

def emit(
    b: Builder,
    ctx: CudaEmitContext,
    *,
    A: int,
    B: int,
    out: int,
    transA: bool = False,
    transB: bool = False,
    name: str = "gemm",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    ta = 1 if bool(transA) else 0
    tb = 1 if bool(transB) else 0
    # 8바이트 페이로드 (transA, transB)
    blob = struct.pack("<ii", ta, tb)

    return emit_resolved(
        b,
        kind="gemm",
        name=name,
        inputs=[A, B],
        outputs=[out],
        kind_id=ctx.Gemm,
        attr_schema=0, 
        attr_blob=blob,
        attrs={"transA": bool(transA), "transB": bool(transB)},
        constraints=constraints,
        hints=hints,
    )

def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,        
    grad_y: int,          
    name: str = "gemm_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD gemm 노드를 바탕으로 BWD gemm들을 추가합니다.
    Y = A @ B (Batch @ Weight^T 형태 등)
    """
    A_vid = fwd_node.inputs[0] # x: [64, 128]
    B_vid = fwd_node.inputs[1] # W: [10, 128] (보통 가중치는 [Out, In])
    
    ta = fwd_node.attrs["transA"]
    tb = fwd_node.attrs["transB"]

    grads = {}

    # 1. grad_A 생성 (dA/dx = dY @ B) -> [64, 128]
    # [64, 10] @ [10, 128] = [64, 128]
    gA_spec = b.values[A_vid].spec
    gA_vid = b.value(f"{name}.dA", gA_spec)
    
    emit(b, ctx, A=grad_y, B=B_vid, out=gA_vid, 
         transA=False, transB=False, name=f"{name}.dA_gemm")
    grads[A_vid] = gA_vid

    # 2. grad_B 생성 (dB/dW = dY^T @ A) -> [10, 128]
    # [10, 64] @ [64, 128] = [10, 128]
    # 💡 교정: dY(grad_y)를 전치하여 앞세우고 x(A_vid)를 뒤에 붙임
    gB_spec = b.values[B_vid].spec
    gB_vid = b.value(f"{name}.dB", gB_spec)
    
    emit(b, ctx, A=grad_y, B=A_vid, out=gB_vid, 
         transA=True, transB=False, name=f"{name}.dB_gemm")
    grads[B_vid] = gB_vid

    return grads