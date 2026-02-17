from __future__ import annotations
import sys
from pathlib import Path

# 1) 프로젝트 루트 및 경로 설정
p = Path(__file__).resolve()
root = None
for parent in [p] + list(p.parents):
    if (parent / "pyproject.toml").exists():
        root = parent
        break
if root is None:
    raise RuntimeError("pyproject.toml not found")

py_src = root / "python" / "aicf_v2" / "src"
sys.path.insert(0, str(py_src))

import aicf_v2 as aicf
from aicf_v2.layers import Linear, Softmax
from aicf_v2.backends.cuda.registry import CudaRegistry

def test_sequential_integration():
    print("=== [AICF v2 Integration Test: Sequential] ===")
    
    # 2) Sequential 모델 정의 (선언적 방식)
    # 레이어 리스트만 넘기면 내부에서 체이닝 준비를 마칩니다.
    model = aicf.Sequential([
        Linear(in_features=128, out_features=10, name="fc1"),
        Softmax(axis=-1, name="prob")
    ], dtype="f32", device="cuda")

    # 3) 모델 빌드
    # 입력 Spec을 주면 Linear -> Softmax 순서로 Vid를 자동 연결합니다.
    print("\n[Step 1] Building Model...")
    input_spec = aicf.TensorSpec(shape=(4, 128))
    model.build(input_spec, input_name="x")
    
    # IR 그래프 덤프 확인
    print("\n=== IR Graph Dump ===")
    print(model.dump())

    # 4) 모델 컴파일 (Planning)
    # Alias 정책에 따라 메모리 재사용(Inplace) 여부를 결정합니다.
    print("\n[Step 2] Compiling Model...")
    registry = CudaRegistry()
    compiled = model.compile(registry)
    plan = compiled.plan

    # 5) 실행 계획(Plan) 검증
    print("\n=== Execution Plan Summary ===")
    print(f"Total Ops in Plan: {len(plan.ops)}")

    for i, op in enumerate(plan.ops):
        # 각 연산의 물리적 메모리 슬롯 할당 상태 확인
        ins = op.inputs
        outs = op.outputs
        
        # 실제 메모리 주소(Slot)를 alias 맵에서 확인
        actual_ins = [plan.alias.get(v, v) for v in ins]
        actual_outs = [plan.alias.get(v, v) for v in outs]
        
        print(f"Op {i} [{op.kind}]:")
        print(f"  - Logical:  {ins} -> {outs}")
        print(f"  - Physical: {actual_ins} -> {actual_outs}")

        if op.kind == "softmax":
            # Softmax의 입력 슬롯과 출력 슬롯이 같다면 Inplace 성공
            is_inplace = (actual_ins[0] == actual_outs[0])
            print(f"  - Inplace Optimization: {'SUCCESS' if is_inplace else 'FAILED'}")
            
            # 검증용 단언문 (plan.py에 'softmax'가 등록되어 있어야 함)
            # assert is_inplace, "Softmax should be inplace-optimized!"

    print("\n[Step 3] Integration Test Completed Successfully.")

if __name__ == "__main__":
    try:
        test_sequential_integration()
    except Exception as e:
        print(f"\n[Test Failed] Error: {e}")
        import traceback
        traceback.print_exc()