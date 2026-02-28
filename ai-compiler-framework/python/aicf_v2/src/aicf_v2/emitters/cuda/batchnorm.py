from __future__ import annotations
import struct
from typing import List, Dict, Any, Sequence

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
    inputs: List[int],
    outputs: List[int],
    eps: float = 1e-5,
    use_running_stats: bool = False,
    name: str = "batchnorm",
    constraints: dict | None = None,
    hints: dict | None = None,
) -> int:
    """BatchNorm Forward 연산을 IR에 기록합니다."""

    # ---- 계약: inputs 의미 ----
    # 기본 계약: inputs[0]=x, inputs[1]=gamma, inputs[2]=beta
    if len(inputs) < 3:
        raise ValueError(f"{name}: expected inputs >= 3 (x, gamma, beta). got {len(inputs)}")

    # ---- 계약: outputs 의미 ----
    # training(use_running_stats=False): outputs 최소 [y, save_mean, save_rstd]
    # inference(use_running_stats=True): outputs 기본 [y]
    if not use_running_stats:
        if len(outputs) < 3:
            raise ValueError(
                f"{name}: training BN expects outputs >= 3 (y, save_mean, save_rstd). got {len(outputs)}"
            )
        out_role = ["y", "save_mean", "save_rstd"]
    else:
        if len(outputs) < 1:
            raise ValueError(f"{name}: inference BN expects outputs >= 1 (y). got {len(outputs)}")
        # inference에서 outputs가 추가로 넘어와도(예: 더미 저장 텐서) role을 맞춰둔다.
        out_role = ["y"]
        if len(outputs) >= 3:
            out_role += ["save_mean", "save_rstd"]

    in_role = ["x", "gamma", "beta"]

    eps_f = float(eps)
    urs = 1 if bool(use_running_stats) else 0
    blob = struct.pack("<fI", eps_f, urs)

    # ---- Static flags ----
    # BN은 외부 관점에서 normalize(+affine)로 취급: elementwise + norm + batchnorm
    # training 통계 계산이 있어도 IS_REDUCE는 FWD에 올리지 않음(오판 방지)
    static = OpFlags.IS_ELEMENTWISE | OpFlags.IS_NORM | OpFlags.IS_BATCHNORM

    # ---- Inplace ----
    # BN은 outputs가 여러 개일 수 있어 inplace를 자동 선호하지 않음.
    # 명시 모드에서만 y(out0) := x(in0) alias를 "선호"로 표시한다.
    inplace_mode = None
    inplace_out_index = None
    inplace_in_index = None
    if constraints:
        inplace_mode = constraints.get("inplace_mode")  # e.g. "y_inplace_only"

    if inplace_mode == "y_inplace_only":
        static |= OpFlags.INPLACE_PREF
        inplace_out_index = 0  # y
        inplace_in_index = 0   # x

    return emit_resolved(
        b,
        kind="batchnorm",
        name=name,
        inputs=list(inputs),
        outputs=list(outputs),
        kind_id=ctx.BatchNormFwd,
        attr_schema=ctx.SCHEMA_BNEP,
        attr_blob=blob,
        attrs={
            "eps": eps_f,
            "use_running_stats": bool(use_running_stats),
            # role 계약(인덱스 하드코딩 제거용)
            "in_role": in_role,
            "out_role": out_role,
            # inplace 계약(어떤 output이 어떤 input과 alias 가능한지)
            "inplace_mode": inplace_mode,
            "inplace_out_index": inplace_out_index,
            "inplace_in_index": inplace_in_index,
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
    name: str = "batchnorm_bwd",
) -> Dict[int, int]:
    """
    최적화된 FWD batchnorm 노드를 역순회하며 BWD 연산을 누적합니다.
    """
    if fwd_node.attrs.get("use_running_stats", False):
        # inference 경로면 통상 training grad를 생성하지 않음
        return {}

    # ---- role 기반 lookup (인덱스 하드코딩 제거) ----
    in_role = fwd_node.attrs.get("in_role", None)
    out_role = fwd_node.attrs.get("out_role", None)

    x = fwd_node.inputs[_role_index(in_role, "x")]
    gamma = fwd_node.inputs[_role_index(in_role, "gamma")]
    beta = fwd_node.inputs[_role_index(in_role, "beta")]

    # training 경로면 save stats가 반드시 있어야 한다.
    if len(fwd_node.outputs) < 3:
        raise ValueError(
            f"{name}: expected fwd outputs >= 3 (y, save_mean, save_rstd). got {len(fwd_node.outputs)}"
        )

    save_mean = fwd_node.outputs[_role_index(out_role, "save_mean")]
    save_rstd = fwd_node.outputs[_role_index(out_role, "save_rstd")]

    x_spec = b.values[x].spec
    stat_spec = b.values[save_mean].spec

    dx = b.value(f"{name}.dx", x_spec)
    dg = b.value(f"{name}.dgamma", stat_spec)
    db = b.value(f"{name}.dbeta", stat_spec)

    # BWD flags:
    # - norm/bn임을 유지
    # - dg/db에 reduce 성격이 강하므로 IS_REDUCE는 BWD에서만 표현 (보수)
    bwd_static = OpFlags.IS_ELEMENTWISE | OpFlags.IS_NORM | OpFlags.IS_BATCHNORM | OpFlags.IS_REDUCE

    emit_resolved(
        b,
        kind="batchnorm_bwd",
        name=name,
        inputs=[x, grad_y, gamma, save_mean, save_rstd],
        outputs=[dx, dg, db],
        kind_id=ctx.BatchNormBwd,
        attr_schema=0,
        attr_blob=b"",
        attrs={
            # 필요하면 bwd도 role을 남겨서 후속 패스에서 활용 가능
            "in_role": ["x", "grad_y", "gamma", "save_mean", "save_rstd"],
            "out_role": ["dx", "dgamma", "dbeta"],
        },
        static_flags=bwd_static,
    )

    return {
        x: dx,
        gamma: dg,
        beta: db,
    }