from __future__ import annotations
import struct
from typing import Dict, Any, Sequence

from ...builder import Builder
from .context import CudaEmitContext
from .base import emit_resolved, OpFlags


def _role_index(role_list: Sequence[str] | None, role: str) -> int:
    if not role_list:
        raise ValueError(f"missing role list while looking for role='{role}'")
    try:
        return list(role_list).index(role)
    except ValueError as e:
        raise ValueError(f"role '{role}' not found in roles={list(role_list)}") from e


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
    """GEMM(General Matrix Multiply) Forward 연산을 IR에 기록합니다.

    Semantics:
      Y = op(A) @ op(B)
      op(X) = X^T if transX else X
    """

    # ---- Role 계약 ----
    in_role = ["A", "B"]
    out_role = ["out"]

    ta = 1 if bool(transA) else 0
    tb = 1 if bool(transB) else 0
    blob = struct.pack("<ii", ta, tb)

    # ---- Static Flags ----
    # GEMM은 fusion/epilogue 패턴의 anchor(루트)로 쓰이는 대표 연산
    static = OpFlags.IS_GEMM_LIKE

    return emit_resolved(
        b,
        kind="gemm",
        name=name,
        inputs=[A, B],
        outputs=[out],
        kind_id=ctx.Gemm,
        attr_schema=0,
        attr_blob=blob,
        attrs={
            "transA": bool(transA),
            "transB": bool(transB),
            "in_role": in_role,
            "out_role": out_role,
        },
        constraints=constraints,
        hints=hints,
        static_flags=static,
    )


def emit_bwd(
    b: Builder,
    ctx: CudaEmitContext,
    fwd_node: Any,
    grad_y: int,
    name: str = "gemm_bwd",
) -> Dict[int, int]:
    """FWD gemm 노드를 바탕으로 BWD gemm들을 추가합니다.

    Forward semantics:
      Y = op(A) @ op(B), op(X)=X^T if transX else X

    Gradients:
      dAeff = dY @ (Beff)^T
      dBeff = (Aeff)^T @ dY
      where Aeff=op(A), Beff=op(B)

    Map back:
      if transA: dA = (dAeff)^T else dA = dAeff
      if transB: dB = (dBeff)^T else dB = dBeff
    """

    in_role = fwd_node.attrs.get("in_role", ["A", "B"])
    A_vid = fwd_node.inputs[_role_index(in_role, "A")]
    B_vid = fwd_node.inputs[_role_index(in_role, "B")]

    ta = bool(fwd_node.attrs.get("transA", False))
    tb = bool(fwd_node.attrs.get("transB", False))

    grads: Dict[int, int] = {}

    # ----- dA -----
    gA_spec = b.values[A_vid].spec
    gA_vid = b.value(f"{name}.dA", gA_spec)

    if not ta:
        # dA = dAeff = dY @ (Beff)^T
        # Beff = op(B) => (Beff)^T is:
        #   if tb==False: B^T  -> transB=True
        #   if tb==True : B    -> transB=False
        emit(
            b,
            ctx,
            A=grad_y,
            B=B_vid,
            out=gA_vid,
            transA=False,
            transB=(not tb),
            name=f"{name}.dA_gemm",
        )
    else:
        # dA = (dAeff)^T = (dY @ (Beff)^T)^T = Beff @ dY^T
        # Beff = op(B) => pass B with transA=tb
        emit(
            b,
            ctx,
            A=B_vid,
            B=grad_y,
            out=gA_vid,
            transA=tb,      # Beff
            transB=True,    # dY^T
            name=f"{name}.dA_gemm",
        )

    grads[A_vid] = gA_vid

    # ----- dB -----
    gB_spec = b.values[B_vid].spec
    gB_vid = b.value(f"{name}.dB", gB_spec)

    if not tb:
        # dB = dBeff = (Aeff)^T @ dY
        # Aeff = op(A) => (Aeff)^T is:
        #   if ta==False: A^T -> transA=True
        #   if ta==True : A   -> transA=False
        emit(
            b,
            ctx,
            A=A_vid,
            B=grad_y,
            out=gB_vid,
            transA=(not ta),
            transB=False,
            name=f"{name}.dB_gemm",
        )
    else:
        # dB = (dBeff)^T = ((Aeff)^T @ dY)^T = dY^T @ Aeff
        # Aeff = op(A) => pass A with transB=ta
        emit(
            b,
            ctx,
            A=grad_y,
            B=A_vid,
            out=gB_vid,
            transA=True,    # dY^T
            transB=ta,      # Aeff
            name=f"{name}.dB_gemm",
        )

    grads[B_vid] = gB_vid

    return grads