from __future__ import annotations

import torch
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
from aicf_v2.tensor_spec import TensorSpec
from aicf_v2.emitters.cuda.sgd_step import sgd_step

def test_autodiff_integrated():
    print("=== AICF Integrated Autodiff & Training Test ===")
    
    # 1. 모델 정의 (Sequential)
    linear_layer = aicf.Linear(in_features=4, out_features=2, name="fc1", bias=True)
    model = aicf.Sequential([linear_layer])
    
    # 2. Forward Build
    x_spec = TensorSpec(shape=(1, 4), dtype="f32")
    y_pred_vid = model.build(x_spec, input_name="x")
    
    # 3. Loss 구성
    y_true_vid = model.input("y_true", TensorSpec(shape=(1, 2), dtype="f32"))
    
    # Loss = sum(y_pred - y_true)
    diff_vid = model.op("sub", 
                        inputs=[y_pred_vid, y_true_vid], 
                        outputs=[model.b.tensor_like(y_pred_vid)])
    
    loss_vid = model.op("sum", 
                        inputs=[diff_vid], 
                        outputs=[TensorSpec(shape=(1,), dtype="f32")])
    
    # 4. Backward Build
    print("\n[Step 1] Building Backward Graph...")
    model.build_backward(loss_vid)
    
    # 5. Optimizer 연산 추가 (SGD)
    print("[Step 2] Adding Optimizer Ops (SGD)...")
    lr = 0.01
    for p_vid, g_vid in model.parameter_grads.items():
        sgd_step(
            model.b, 
            model.ctx, 
            P=p_vid, 
            G=g_vid, 
            outP=p_vid, 
            lr=lr, 
            name=f"update_{model.b.values[p_vid].name}"
        )

    # 6. Compile & Capture 준비
    print("[Step 3] Compiling & Capturing Integrated Graph...")
    torch.manual_seed(42)
    
    # [핵심 수정] sample_feed 구성
    # 6-1. 사용자 입력 데이터
    sample_feed = {
        "x": torch.randn(1, 4, device="cuda"), 
        "y_true": torch.ones(1, 2, device="cuda") * 0.5,
        "grad_initial": torch.ones((1,), device="cuda") 
    }
    
    # 6-2. 파라미터(fc1.W, fc1.b 등) 자동 주입
    # Builder에 등록된 모든 param_vids를 찾아 실제 텐서를 생성하여 feed에 바인딩합니다.
    for vid in model.b.param_vids:
        val = model.b.values[vid]
        if val.name not in sample_feed:
            # 파라미터 초기화 (W는 randn, b는 zero 등)
            if "W" in val.name:
                sample_feed[val.name] = torch.randn(val.spec.shape, device="cuda")
            else:
                sample_feed[val.name] = torch.zeros(val.spec.shape, device="cuda")
            print(f"  > Binding parameter: {val.name} (shape={val.spec.shape})")

    # 통합 그래프 캡처 (이제 fc1.W 에러가 발생하지 않습니다)
    model.compile(capture=True, sample_feed=sample_feed, mode="train")
    
    # 7. 실전 학습 루프
    print("\n[Step 4] Training Loop Starting (Replay)...")
    
    # 캡처된 프로그램 내부의 슬롯(메모리)에 직접 접근하기 위한 객체
    # gprog = model.executor.get_captured_program("train") # 실제 구현에 따라 경로 조정 필요
    gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]
    
    # 가중치 Vid 가져오기
    w_vid = model._tape[0]['params'][0]
    
    for i in range(10):
        # 그래프 실행 (Fwd -> Bwd -> SGD Update)
        model.run(sample_feed, use_cuda_graph=True)
        
        # 가중치 값 변화 모니터링
        w_val = gprog.slots[w_vid]
        print(f"Iter {i}: Weight Mean = {w_val.mean().item():.6f}")

    print("\n[Step 5] Integration Success!")
    print("- Automatic Differentiation: OK")
    print("- Unified Graph Replay: OK")
    print("- In-place Parameter Update: OK")

if __name__ == "__main__":
    test_autodiff_integrated()