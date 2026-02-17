from __future__ import annotations
import torch
import sys
from pathlib import Path

# 1) 프로젝트 루트 및 경로 설정
p = Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
sys.path.insert(0, str(root / "python" / "aicf_v2" / "src"))

import aicf_v2 as aicf
from aicf_v2.tensor_spec import TensorSpec
from aicf_v2.layers.mse import MSELoss

# 신규 도입된 옵티마이저 임포트
# (위에서 정의한 optimizers/sgd.py 또는 optimizers/adam.py 사용)
from aicf_v2.optimizers.sgd import SGD
from aicf_v2.optimizers.adam import Adam

def test_autodiff_integrated(use_adam: bool = True):
    optimizer_name = "Adam" if use_adam else "SGD"
    print(f"=== AICF Integrated Autodiff & Training Test ({optimizer_name}) ===")
    
    # 1. 모델 정의 (Sequential)
    linear_layer = aicf.Linear(in_features=4, out_features=2, name="fc1", bias=True)
    model = aicf.Sequential([linear_layer])
    
    # 2. Forward Build
    x_spec = TensorSpec(shape=(1, 4), dtype="f32")
    y_pred_vid = model.build(x_spec, input_name="x")
    
    # 3. Loss Build (정식 MSELoss 레이어 사용)
    y_true_vid = model.input("y_true", TensorSpec(shape=(1, 2), dtype="f32"))
    loss_vid = model.add(MSELoss(reduction="mean", name="loss"), y_pred_vid, y_true_vid)
    
    # 4. Backward Build (미분 그래프 생성)
    print("\n[Step 1] Building Backward Graph...")
    model.build_backward(loss_vid)
    
    # 5. Optimizer Step (전략 매니저 사용)
    # 이제 수동 루프 없이 optimizer.step() 하나로 업데이트 연산이 그래프에 통합됩니다.
    print(f"[Step 2] Adding Optimizer Ops ({optimizer_name})...")
    if use_adam:
        optimizer = Adam(model, lr=0.001)
    else:
        optimizer = SGD(model, lr=0.01)
        
    optimizer.step()

    # 6. Compile & Capture
    print("[Step 3] Compiling & Capturing Integrated Graph...")
    torch.manual_seed(42)
    
    # 6-1. 기본 피드 데이터
    sample_feed = {
        "x": torch.randn(1, 4, device="cuda"), 
        "y_true": torch.ones(1, 2, device="cuda") * 0.5,
        "grad_initial": torch.ones((1,), device="cuda") 
    }
    
    # 6-2. Adam 사용 시 편향 보정치(bc1, bc2) 초기값 주입
    if use_adam:
        # t=1 시점의 초기값 (beta1^1, beta2^1)
        sample_feed["adam.bc1"] = torch.tensor([0.9], device="cuda")
        sample_feed["adam.bc2"] = torch.tensor([0.999], device="cuda")
    
    # 통합 그래프 캡처 (Fwd + Loss + Bwd + Opt)
    model.compile(capture=True, sample_feed=sample_feed, mode="train")
    
    # 7. 실전 학습 루프 (Replay)
    print(f"\n[Step 4] Training Loop Starting (Replay - {optimizer_name})...")
    
    # 가중치 Vid 추출 (fc1.W)
    w_vid = model._tape[0]['params'][0]
    # 캡처된 결과가 저장된 슬롯 확인을 위해 그래프 객체 획득
    gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]
    
    for i in range(10):
        # [핵심] Adam 사용 시 매 스텝 bc1, bc2를 업데이트해서 feed로 전달 가능
        # (현재는 고정값 테스트, 이후 StepInc 레이어로 자동화 가능)
        model.run(sample_feed, use_cuda_graph=True)
        
        # 가중치 값 변화 모니터링
        w_val = gprog.slots[w_vid]
        print(f"Iter {i}: Weight Mean = {w_val.mean().item():.6f}")

    print(f"\n[Step 5] {optimizer_name} Integration Success!")

if __name__ == "__main__":
    # SGD 테스트
    test_autodiff_integrated(use_adam=False)
    print("-" * 50)
    # Adam 테스트 (adam_step 커널이 등록되어 있어야 함)
    # test_autodiff_integrated(use_adam=True)