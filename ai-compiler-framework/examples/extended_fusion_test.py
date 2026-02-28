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
from aicf_v2.compile.passes.pipeline import optimize_ir

def snapshot_ops(b) -> List[Tuple[str, str, List[int], List[int]]]:
    """IR 상태를 간결하게 스냅샷 찍기"""
    return [(op.kind, op.name, list(op.inputs), list(op.outputs)) for op in b.ops]

def main():
    print("=== [1. 모델 및 그래프 빌드] ===")
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
    ])
    
    # 입력 및 Forward 실행
    x_spec = aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda")
    y_vid = model.build(x_spec, input_name="x")

    # Loss 추가 (Scalar Reduction)
    from aicf_v2.emitters.cuda.reduce_sum import emit as emit_reduce
    final_loss_vid = model.b.value("loss", aicf.TensorSpec(shape=(1,), dtype="f32", device="cuda"))
    emit_reduce(model.b, model.ctx, x=y_vid, out=final_loss_vid, axis=0)

    b = model.b
    print(f"Initial Ops: {len(b.ops)}")

    print("\n=== [2. Forward Graph Optimization (Fusion)] ===")
    before_fusion = snapshot_ops(b)
    
    # 퓨전 패스 실행
    optimize_ir(b)
    
    after_fusion = snapshot_ops(b)
    
    # Fusion 검증
    fused_ops = [op for op in b.ops if op.kind == "fused_gemm_bias_relu"]
    assert len(fused_ops) == 1, "Fusion failed: fused_gemm_bias_relu not found."
    print(f"✅ Fusion Success: {fused_ops[0].name} created.")

    print("\n=== [3. Autograd: Backward Graph Generation] ===")
    # 역전파 실행: Loss로부터 입력을 향해 그래디언트 전파
    # fused_node.bwd_emit_fn 이 여기서 실제로 호출됨
    grad_map = model.backward(final_loss_vid)
    
    print(f"Ops after Backward: {len(b.ops)}")
    for i, op in enumerate(b.ops):
        if "grad" in op.name or "bwd" in op.name or "db" in op.name:
            print(f"Bwd-Op[{i:02d}] kind={op.kind:<20} name={op.name:<25} in={op.inputs}")

    print("\n=== [4. 정밀 검증 (Verification)] ===")
    
    # 1. Fused 노드가 3개의 입력을 가지는지 (X, W, Bias)
    f_op = fused_ops[0]
    assert len(f_op.inputs) == 3, f"Fused op should have 3 inputs, got {len(f_op.inputs)}"

    # 2. 역전파가 입력 0(x)까지 도달했는지 확인
    x_vid = 0 # 첫 번째 입력 Vid
    assert x_vid in grad_map, "Gradient did not reach to input 'x'"
    print(f"✅ Autograd Success: Gradient reached to Vid[{x_vid}]")

    # 3. NOP 노드들이 역전파에 영향을 주지 않았는지 확인
    nop_ops = [op for op in b.ops if op.kind == "nop"]
    assert len(nop_ops) == 2, f"Expected 2 NOP nodes, got {len(nop_ops)}"
    print(f"✅ Graph Integrity: {len(nop_ops)} nodes safely neutralized.")

    print("\n[최종 IR 결과 요약]")
    for i, op in enumerate(b.ops):
        print(f"[{i:02d}] {op.kind:<22} | {op.name}")

    print("\n🎉 모든 확장 테스트 통과!")

if __name__ == "__main__":
    main()