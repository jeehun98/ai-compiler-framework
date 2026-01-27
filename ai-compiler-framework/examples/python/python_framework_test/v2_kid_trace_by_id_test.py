# examples/python/python_framework_test/v2_kid_trace_by_id_test.py
from __future__ import annotations

import sys
from pathlib import Path
import time
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]  # .../examples/python
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))


from aicf_fw.core_v2 import trace_ir, dump_ir, dump_lowered, dump_plan
from aicf_fw.core_v2.ops import (
    sym_tensor,
    linear,
    relu,
    save,
    mse_grad,
    linear_bwd,
    relu_bwd,
    sgd_step,
)
from aicf_fw.core_v2.lower import lower_to_backend_ops
from aicf_fw.core_v2.plan import build_binding_plan, apply_kernel_decisions_stageB
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions
from aicf_fw.core_v2.rewrites.stageC_fuse_epilogue import stageC_fuse_gemm_epilogue

# NEW: OpAttrs dump
from aicf_fw.core_v2.op_attrs.registry import build_op_attr


def tf32_off():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    try:
        torch.set_float32_matmul_precision("highest")
    except Exception:
        pass


def require_contains(hay: str, needle: str):
    if needle not in hay:
        raise AssertionError(f"expected to find '{needle}' in trace, but not found.")


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)
    return p


def dump_text(path: Path, text: str):
    path.write_text(text, encoding="utf-8")


def _dtype_str(v) -> str:
    dt = getattr(v, "dtype", "torch.float32")
    return str(dt)


def _is_f16(s: str) -> bool:
    s = str(s)
    return ("float16" in s) or ("torch.float16" in s) or ("Half" in s)


def _pick_kernel_id_fallback(op: str, in_dtypes: list[str], out_dtypes: list[str], attrs: dict) -> str | None:
    """
    Safety-net: StageB가 vec2/half2 업그레이드만 하고 기본 kid를 안 채우는 경우를 막기 위해
    테스트 코드에서 "빈 kid 채우기"를 보장한다.

    ⚠️ 여기 mapping은 register_all.cpp의 kid 문자열과 1:1로 맞춰야 한다.
    """
    op = str(op).strip().lower()
    in0 = in_dtypes[0] if in_dtypes else ""
    out0 = out_dtypes[0] if out_dtypes else ""

    if op == "gemm":
        return "gemm_f16_tc_wmma_out_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "gemm_f32_naive_v0"

    if op == "gemm_epilogue":
        # 현재 구현한 epilogue kid 기준 (bias+relu fused)
        return "gemm_bias_relu_f16_tc_wmma_out_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "gemm_bias_relu_f32_naive_v0"

    if op == "bias_add":
        return "bias_add_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "bias_add_f32_v0"

    if op == "add":
        return "add_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "add_f32_v0"

    if op == "relu":
        return "relu_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "relu_f32_v0"

    if op == "mse_grad":
        return "mse_grad_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "mse_grad_f32_v0"

    if op == "relu_bwd":
        return "relu_bwd_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "relu_bwd_f32_v0"

    if op == "reduce_sum":
        # reduce_sum_keep_lastdim = bias grad rowsum
        if _is_f16(out0):
            return "reduce_sum_keep_lastdim_f16_v0"
        if _is_f16(in0):
            return "reduce_sum_keep_lastdim_f16_to_f32_v0"
        return "reduce_sum_keep_lastdim_f32_v0"

    if op == "sgd_step":
        return "sgd_step_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "sgd_step_f32_v0"

    if op in ("copy", "copy_saved", "copy_aux"):
        return "copy_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "copy_f32_v0"

    if op == "grad_zero":
        return "grad_zero_v0"
    if op == "adam_step":
        return "adam_step_f32_v0"
    if op == "step_inc":
        return "step_inc_v0"
    if op in ("bias_corr", "biascorr"):
        return "bias_corr_v0"
    if op == "layernorm_fwd":
        return "layernorm_fwd_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "layernorm_fwd_f32_v0"
    if op == "layernorm_bwd":
        return "layernorm_bwd_f16_v0" if (_is_f16(in0) or _is_f16(out0)) else "layernorm_bwd_f32_v0"
    if op == "batchnorm_fwd":
        return "batchnorm_fwd_f16_v0"
    if op == "batchnorm_bwd":
        return "batchnorm_bwd_f16_v0"

    return None


def _fill_missing_kernel_ids(ir, lowered: list[dict]) -> None:
    """
    lowered 내 kernel_id가 None인 항목을 전부 채운다.
    (StageA/StageB의 구현 세부가 바뀌어도 테스트가 깨지지 않게 하는 안전장치)
    """
    for it in lowered:
        if it.get("kernel_id", None) is not None:
            continue

        op = str(it.get("op", it.get("kind", ""))).strip().lower()
        in_vids = [int(x) for x in it.get("inputs", [])]
        out_vids = [int(y) for y in it.get("outputs", [])]
        attrs = dict(it.get("attrs", {}) or {})

        in_dtypes = [_dtype_str(ir.values[v]) for v in in_vids]
        out_dtypes = [_dtype_str(ir.values[v]) for v in out_vids]

        kid = _pick_kernel_id_fallback(op, in_dtypes, out_dtypes, attrs)
        if kid is None:
            raise RuntimeError(
                f"[test] cannot fill kernel_id: op={op} inputs={in_vids} outputs={out_vids} attrs={attrs} "
                f"in_dtypes={in_dtypes} out_dtypes={out_dtypes}"
            )
        it["kernel_id"] = kid


# ----------------------------
# NEW: dump op attrs
# ----------------------------
def dump_op_attrs(ir, lowered: list[dict], name: str = "") -> str:
    """
    loweredops -> OpAttr 변환 결과를 사람이 보기 쉬운 텍스트로 덤프.
    목적: 중간 산출물/디버깅 (아직 compose/selection 없음)
    """
    lines: list[str] = []
    title = f"=== OpAttrs({name}) ===" if name else "=== OpAttrs ==="
    lines.append(title)
    lines.append(f"ops: {len(lowered)}")
    lines.append("")

    # TensorDesc.from_any()가 torch.Tensor를 처리하므로 ir.values 그대로 전달
    value_descs = ir.values

    for i, lop in enumerate(lowered):
        # build_op_attr가 dict의 kind/name/op_kind를 보도록 설계됐을 수 있으니
        # lowered dict가 {"op": "..."} 형태면 kind를 보강
        if isinstance(lop, dict) and ("kind" not in lop) and ("op" in lop):
            lop_view = dict(lop)
            lop_view["kind"] = lop_view.get("op")
        else:
            lop_view = lop

        oa = build_op_attr(lop_view, value_descs, op_id=i)

        kid = (
            oa.kid
            or (lop.get("kernel_id") if isinstance(lop, dict) else None)
            or (lop.get("kid") if isinstance(lop, dict) else None)
        )

        lines.append(
            f"  #{i:03d} {oa.op_kind:<10} sig={oa.sig or '-':<10} "
            f"in={oa.inputs} out={oa.outputs} kid={kid}"
        )

        if oa.params:
            lines.append(f"       params: {oa.params}")
        if oa.layout:
            lines.append(f"       layout: {oa.layout}")

        in0s = oa.shapes.get("in0", None)
        out0s = oa.shapes.get("out0", None)
        in0d = oa.dtypes.get("in0", None)
        out0d = oa.dtypes.get("out0", None)
        if in0s is not None or out0s is not None:
            lines.append(f"       io0: in0={in0s}/{in0d}  out0={out0s}/{out0d}")

        lines.append("")

    return "\n".join(lines)


def main():
    tf32_off()
    torch.manual_seed(0)

    device = torch.device("cuda:0")
    dtype = torch.float16
    B, D = 64, 8  # vec2/half2 조건

    x = torch.randn(B, D, device=device, dtype=dtype)
    t = torch.randn(B, D, device=device, dtype=dtype)

    W0 = torch.randn(D, D, device=device, dtype=dtype)
    b0 = torch.randn(D, device=device, dtype=dtype)
    W1 = torch.randn(D, D, device=device, dtype=dtype)
    b1 = torch.randn(D, device=device, dtype=dtype)

    lr = 1e-2

    def build():
        sx = sym_tensor(name="x", shape=(B, D), dtype=dtype, device=device)
        st = sym_tensor(name="t", shape=(B, D), dtype=dtype, device=device)

        sW0 = sym_tensor(name="0.W", shape=(D, D), dtype=dtype, device=device)
        sb0 = sym_tensor(name="0.b", shape=(D,), dtype=dtype, device=device)
        sW1 = sym_tensor(name="2.W", shape=(D, D), dtype=dtype, device=device)
        sb1 = sym_tensor(name="2.b", shape=(D,), dtype=dtype, device=device)

        lin0 = linear(sx, sW0, sb0, name="lin0_out")
        r0 = relu(lin0, name="relu0_out")
        r0s = save(r0, name="relu0_saved")
        lin1 = linear(r0, sW1, sb1, name="lin1_out")
        dY = mse_grad(lin1, st, name="dY")

        d_r0, dW1, db1 = linear_bwd(
            r0, sW1, dY,
            bias=True,
            dx_name="d_relu0_out",
            dW_name="d_2.W",
            db_name="d_2.b",
        )
        d_lin0 = relu_bwd(d_r0, r0s, name="d_lin0_out")
        _dx, dW0, db0 = linear_bwd(
            sx, sW0, d_lin0,
            bias=True,
            dx_name="d_x",
            dW_name="d_0.W",
            db_name="d_0.b",
        )

        sgd_step(sW0, dW0, lr=lr)
        sgd_step(sb0, db0, lr=lr)
        sgd_step(sW1, dW1, lr=lr)
        sgd_step(sb1, db1, lr=lr)

    art_root = ensure_dir(Path("artifacts") / f"{now_tag()}_v2_kid_trace_by_id_test")

    ir = trace_ir(build, name="v2_kid_trace_by_id_test")
    ir_txt = dump_ir(ir)
    print(ir_txt)
    dump_text(art_root / "20_ir.txt", ir_txt)

    lowered = lower_to_backend_ops(ir)                    # Stage A
    lowered = apply_kernel_decisions_stageB(ir, lowered)  # Stage B (vec2/half2 + (optional) rewrite)

    lowered = stageC_fuse_gemm_epilogue(ir, lowered)


    # StageB가 업그레이드만 하고 기본 kid를 안 채우는 케이스 대비
    _fill_missing_kernel_ids(ir, lowered)

    lowered_txt = dump_lowered(lowered, name="v2_kid_trace_by_id_test")
    print(lowered_txt)
    dump_text(art_root / "30_lowered.txt", lowered_txt)

    # NEW: OpAttrs dump (중간 산출물)
    op_attrs_txt = dump_op_attrs(ir, lowered, name="v2_kid_trace_by_id_test")
    print(op_attrs_txt)
    dump_text(art_root / "31_op_attrs.txt", op_attrs_txt)

    plan = build_binding_plan(ir)
    plan_txt = dump_plan(plan, name="v2_kid_trace_by_id_test")
    print(plan_txt)
    dump_text(art_root / "40_plan.txt", plan_txt)

    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        opts=ExecOptions(debug=False, require_kernel_id=True),
    )

    params = {"0.W": W0, "0.b": b0, "2.W": W1, "2.b": b1}

    ex.trace_reset()
    ex.trace_enable(True)
    ex.run(inputs={"x": x, "t": t}, params=params, reuse_static=True)
    ex.trace_enable(False)

    trace = "\n".join([str(s) for s in ex.trace_get()])
    print("=== TRACE ===")
    print(trace)
    dump_text(art_root / "50_runtime_trace.txt", trace)

    # 기대: StageB half2 업그레이드가 실제 실행에 반영
    require_contains(trace, "sgd_step_f16_half2_v0")

    print(f"[OK] artifacts dumped to: {art_root}")
    print("OK")


if __name__ == "__main__":
    main()
