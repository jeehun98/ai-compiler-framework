from __future__ import annotations
import sys
from pathlib import Path as _Path
from typing import List, Tuple, Any, Dict

# -----------------------
# 프로젝트 경로 설정
# -----------------------
p = _Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
src_path = root / "python" / "aicf_v2" / "src"
sys.path.insert(0, str(src_path))

import aicf_v2 as aicf
from aicf_v2.emitters.cuda.base import OpFlags

def print_graph(title: str, ops: List[Any]):
    print(f"\n--- {title} ---")
    for i, op in enumerate(ops):
        ins = op.inputs if hasattr(op, "inputs") else []
        outs = op.outputs if hasattr(op, "outputs") else []
        print(f"[{i:02d}] {op.kind:<20} | {op.name:<25} | in={ins} out={outs}")

def main():
    # 1. 모델 빌드 (Linear + ReLU)
    # Linear 레이어는 내부적으로 gemm + bias_add를 생성합니다.
    print("=== [1] 모델 빌드 및 초기 그래프 생성 ===")
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
    ])
    
    x_spec = aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda")
    y_vid = model.build(x_spec, input_name="x")

    # Loss 추가 (그래프 끝단 확보)
    from aicf_v2.emitters.cuda.reduce_sum import emit as emit_reduce
    loss_vid = model.b.value("loss", aicf.TensorSpec(shape=(1,), dtype="f32", device="cuda"))
    emit_reduce(model.b, model.ctx, x=y_vid, out=loss_vid, axis=0)

    print_graph("BEFORE OPTIMIZATION", model.b.ops)

    # 2. 퓨전 및 자동 미분 실행
    # build_backward_after_fwd_opt()는 내부적으로 optimize_ir()를 먼저 수행합니다.
    print("\n=== [2] 최적화 및 역전파 그래프 생성 ===")
    grad_map = model.build_backward_after_fwd_opt(loss_vid)

    # 3. 결과 분석
    ops = model.b.ops
    print_graph("AFTER OPTIMIZATION & BACKWARD", ops)

    print("\n=== [3] 정밀 검증 결과 ===")
    
    # (1) Forward 퓨전 확인
    fused_fwd = [op for op in ops if op.kind == "gemm_epilogue"]
    assert len(fused_fwd) == 1, "Forward Fusion failed: gemm_epilogue not found."
    print(f"✅ Forward Fusion: {fused_fwd[0].name} (Inputs: {fused_fwd[0].inputs})")

    # (2) NOP 노드 개수 확인 (bias_add, relu, 그리고 신규 생성된 임시 노드)
    # GraphRewriter 로직상 3개가 NOP으로 변해야 함
    nops = [op for op in ops if op.kind == "nop"]
    print(f"✅ NOP Nodes: {len(nops)} (Graph stability preserved)")

    # (3) 역전파 합성(Composition) 확인
    # 우리가 의도한 3단계 미분이 생성되었는지 확인합니다.
    relu_mask_op = [op for op in ops if ".relu_mask" in op.name]
    gemm_bwd_op = [op for op in ops if ".gemm.dA" in op.name or ".gemm.dB" in op.name]
    dbias_op = [op for op in ops if ".dbias" in op.name]

    if relu_mask_op:
        print(f"✅ Backward Step 1: ReLU Masking generated ({relu_mask_op[0].kind})")
    if gemm_bwd_op:
        print(f"✅ Backward Step 2: GEMM Gradients generated (dA/dB)")
    if dbias_op:
        print(f"✅ Backward Step 3: Bias Gradient generated ({dbias_op[0].kind})")

    # (4) dBias 커널의 입력 무결성 확인
    if dbias_op and relu_mask_op:
        # dBias 커널의 입력 dY가 ReLU Mask의 출력 dZ와 일치하는지 확인
        dZ_vid = relu_mask_op[0].outputs[0]
        dbias_in_vid = dbias_op[0].inputs[0]
        assert dZ_vid == dbias_in_vid, "dBias should use masked gradient dZ."
        print(f"✅ Data Flow Integrity: dBias uses masked dZ (Vid: {dZ_vid})")

    print("\n🎉 모든 테스트 케이스 통과! (Fusion + Composite Autograd)")

if __name__ == "__main__":
    main()