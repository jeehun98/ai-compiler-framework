from __future__ import annotations

import sys
from pathlib import Path as _Path
from typing import List, Tuple, Any, Dict

# -----------------------
# project path bootstrap
# -----------------------
p = _Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
build_lib_path = root / "build" / "python" / "aicf_cuda"
src_path = root / "python" / "aicf_v2" / "src"
if build_lib_path.exists():
    sys.path.insert(0, str(build_lib_path))
sys.path.insert(0, str(src_path))

import aicf_v2 as aicf

# ✅ 네가 옮긴 위치 기준
from aicf_v2.compile.passes.pipeline import optimize_ir, optimize_plan


def snapshot_ops(b) -> List[Tuple[str, int, str, List[int], List[int]]]:
    """(kind, kind_id, name, inputs, outputs) 스냅샷"""
    snap = []
    for op in b.ops:
        snap.append((
            getattr(op, "kind", None),
            int(getattr(op, "kind_id", -1) or -1),
            getattr(op, "name", ""),
            list(getattr(op, "inputs", []) or []),
            list(getattr(op, "outputs", []) or []),
        ))
    return snap


def print_diff(before, after):
    print("\n[IR DIFF]")
    n = max(len(before), len(after))
    for i in range(n):
        b = before[i] if i < len(before) else None
        a = after[i] if i < len(after) else None
        if b == a:
            continue
        print(f"- Op[{i:02d}] BEFORE: {b}")
        print(f"+ Op[{i:02d}] AFTER : {a}")


def build_value_to_producer(b) -> Dict[int, int]:
    v2p: Dict[int, int] = {}
    for i, op in enumerate(b.ops):
        for vid in getattr(op, "outputs", []) or []:
            v2p[vid] = i
    return v2p


def main():
    # 1) 모델 빌드
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
    ])
    x_spec = aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda")
    y_vid = model.build(x_spec, input_name="x")

    # Loss 추가 (그래프 끝단 consumer 확보)
    from aicf_v2.emitters.cuda.reduce_sum import emit as emit_reduce
    final_loss_vid = model.b.value("final_loss", aicf.TensorSpec(shape=(1,), dtype="f32", device="cuda"))
    emit_reduce(model.b, model.ctx, x=y_vid, out=final_loss_vid, axis=0)

    b = model.b

    # 2) optimize_ir 적용 전 스냅샷
    before = snapshot_ops(b)

    print("\n[BEFORE optimize_ir]")
    for i, (kind, kid, name, ins, outs) in enumerate(before):
        print(f"Op[{i:02d}] kind={kind:<18} kid={kid:<4} name={name:<22} in={ins} out={outs}")

    print("\n[DEBUG Flags Check]")
    from aicf_v2.emitters.cuda.base import OpFlags
    for i, op in enumerate(b.ops):
        flags = getattr(op, "static_flags", 0)
        print(f"Op[{i:02d}] kind={op.kind} static_flags={bin(flags)}")

    # 3) ✅ pipeline.optimize_ir 호출 (네가 옮긴 패스가 실제로 실행되는지)
    optimize_ir(b)

    after = snapshot_ops(b)

    print("\n[AFTER optimize_ir]")
    for i, (kind, kid, name, ins, outs) in enumerate(after):
        print(f"Op[{i:02d}] kind={kind:<18} kid={kid:<4} name={name:<22} in={ins} out={outs}")

    # 4) diff 출력
    print_diff(before, after)

    # 5) 핵심 assert: fusion이 실제로 발생했는지
    fused = [(i, op) for i, op in enumerate(b.ops) if getattr(op, "kind", None) == "fused_gemm_bias_relu"]
    assert len(fused) == 1, f"Expected 1 fused op after optimize_ir, got {len(fused)}"
    fi, fop = fused[0]
    assert int(getattr(fop, "kind_id", -1) or -1) == 100, f"Fused kind_id expected 100, got {getattr(fop,'kind_id',None)}"
    assert hasattr(fop, "bwd_emit_fn"), "Fused op missing bwd_emit_fn (hook not injected)"

    # 6) 그래프가 끊기지 않았는지 최소 체크: reduce_sum 입력 producer가 존재
    rs = [(i, op) for i, op in enumerate(b.ops) if getattr(op, "kind", None) == "reduce_sum"]
    assert len(rs) == 1, f"Expected 1 reduce_sum op, got {len(rs)}"
    _, rsop = rs[0]
    rs_in_vid = rsop.inputs[0]
    v2p = build_value_to_producer(b)
    assert rs_in_vid in v2p, f"reduce_sum input vid={rs_in_vid} has no producer (graph broken)"

    # 7) optimize_plan은 현재 identity(pass-through)여도 호출 smoke 체크만
    #    (plan 생성 경로가 있다면, 여기서 plan 만들어서 optimize_plan 호출해도 됨)
    #    지금은 함수 존재/임포트/호출 가능 여부만 체크
    try:
        optimize_plan  # just referenced
        print("\n✅ optimize_plan import OK (currently pass-through).")
    except Exception as e:
        raise AssertionError(f"optimize_plan import failed: {e}")

    print("\n✅ PASS: 기존 테스트의 fusion 로직이 pipeline.optimize_ir로 정상 이관되어 동작함.")


if __name__ == "__main__":
    main()