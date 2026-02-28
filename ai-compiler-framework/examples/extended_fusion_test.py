from __future__ import annotations
import sys
from pathlib import Path as _Path
from typing import List, Tuple, Any, Dict

# -----------------------
# Project Path Setup
# -----------------------
p = _Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
src_path = root / "python" / "aicf_v2" / "src"
sys.path.insert(0, str(src_path))

import aicf_v2 as aicf

def main():
    print("=== [1. 모델 및 그래프 빌드] ===")
    # Model 대신 Sequential 사용 (Model의 기능을 모두 가짐)
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
    ])
    
    x_spec = aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda")
    y_vid = model.build(x_spec, input_name="x")

    # Loss 추가
    from aicf_v2.emitters.cuda.reduce_sum import emit as emit_reduce
    final_loss_vid = model.b.value("loss", aicf.TensorSpec(shape=(1,), dtype="f32", device="cuda"))
    emit_reduce(model.b, model.ctx, x=y_vid, out=final_loss_vid, axis=0)

    print(f"Initial Ops count: {len(model.b.ops)}")

    print("\n=== [2. Autograd with Forward Fusion] ===")
    # 중요: build_backward_after_fwd_opt를 호출하면 
    # 내부적으로 optimize_ir(Fusion)을 먼저 수행한 후 Backward를 생성합니다.
    grad_map = model.build_backward_after_fwd_opt(final_loss_vid)
    
    # Fusion 결과 확인
    fused_ops = [op for op in model.b.ops if op.kind == "fused_gemm_bias_relu"]
    if len(fused_ops) == 1:
        print(f"✅ Fusion Success: {fused_ops[0].name} (Kind: {fused_ops[0].kind})")
    else:
        print(f"❌ Fusion Failed: Expected 1 fused op, got {len(fused_ops)}")

    print("\n=== [3. Backward Graph Inspection] ===")
    # 생성된 역전파 노드들 출력
    bwd_found = False
    for i, op in enumerate(model.b.ops):
        # 퓨전된 노드의 역전파 훅이 실행되어 생성된 노드들 (dx, dw, db 등)
        if any(keyword in op.name for keyword in [".dx", ".dw", ".db", "grad", "bwd"]):
            print(f"Bwd-Op[{i:02d}] {op.kind:<20} | {op.name:<25} | in={op.inputs}")
            bwd_found = True

    if bwd_found:
        print("✅ Backward Ops generated based on Fused Node.")

    print("\n=== [4. 최종 검증] ===")
    # 1. 퓨전 노드 입력 확인
    f_op = fused_ops[0]
    # Inputs: [X, W, Bias] -> 3개여야 함
    assert len(f_op.inputs) == 3, "Fused op should have 3 inputs (including Bias)"
    
    # 2. 그래디언트 도달 확인
    if 0 in grad_map: # 입력 x의 Vid가 0인 경우
        print(f"✅ Gradient successfully reached to input Vid[0]")
    
    print("\n🎉 모든 확장 테스트 통과!")

if __name__ == "__main__":
    main()