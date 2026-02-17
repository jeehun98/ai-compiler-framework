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
from aicf_v2.layers.relu import ReLU
from aicf_v2.optimizers.adam import Adam

def test_mlp_integrated():
    print("=== AICF MLP (2-Layer) Training Integration Test ===")
    
    # 설정: 4(In) -> 8(Hidden) -> 2(Out)
    torch.manual_seed(42)
    in_dim, hidden_dim, out_dim = 4, 8, 2
    lr = 0.01

    # 1. 모델 정의 (Sequential MLP)
    # Linear -> ReLU -> Linear 구조
    model = aicf.Sequential([
        aicf.Linear(in_features=in_dim, out_features=hidden_dim, name="fc1"),
        aicf.ReLU(name="relu1"),
        aicf.Linear(in_features=hidden_dim, out_features=out_dim, name="fc2")
    ])
    
    # 2. Forward Build
    x_spec = TensorSpec(shape=(1, in_dim), dtype="f32")
    y_pred_vid = model.build(x_spec, input_name="x")
    
    # 3. Loss Build
    y_true_vid = model.input("y_true", TensorSpec(shape=(1, out_dim), dtype="f32"))
    loss_vid = model.add(MSELoss(reduction="mean", name="loss"), y_pred_vid, y_true_vid)
    
    # 4. Backward Build
    print("\n[Step 1] Building MLP Backward Graph...")
    model.build_backward(loss_vid)
    
    # 5. Optimizer Step (Adam)
    print("[Step 2] Adding Adam Optimizer Ops...")
    optimizer = Adam(model, lr=lr)
    optimizer.step()

    # 6. Compile & Capture
    print("[Step 3] Compiling & Capturing MLP Graph...")
    sample_x = torch.randn(1, in_dim, device="cuda")
    sample_y = torch.ones(1, out_dim, device="cuda") * 0.7
    
    sample_feed = {
        "x": sample_x,
        "y_true": sample_y,
        "grad_initial": torch.ones((1,), device="cuda"),
        "adam.bc1": torch.tensor([0.9], device="cuda"),
        "adam.bc2": torch.tensor([0.999], device="cuda")
    }
    
    model.compile(capture=True, sample_feed=sample_feed, mode="train")
    
    # 7. Training Loop (Replay)
    print("\n[Step 4] MLP Training Loop Starting...")
    
    # [수정] model.b.values는 list이므로 enumerate를 사용하여 순회하며 vid(index)를 찾습니다.
    def get_vid_by_name(name):
        for vid, val in enumerate(model.b.values):
            if val.name == name: 
                return vid
        return None

    w1_vid = get_vid_by_name("fc1.W")
    w2_vid = get_vid_by_name("fc2.W")
    
    if w1_vid is None or w2_vid is None:
        raise RuntimeError("가중치 Vid를 찾을 수 없습니다. 파라미터 이름을 확인하세요.")

    # 캡처된 그래프 객체 획득
    gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]
    
    for i in range(15):
        # 매 스텝 run (CUDA Graph Replay)
        model.run(sample_feed, use_cuda_graph=True)
        
        # 가중치 슬롯에서 현재 값 추출
        w1_val = gprog.slots[w1_vid]
        w2_val = gprog.slots[w2_vid]
        
        if i % 3 == 0:
            print(f"Iter {i:2d} | fc1.W Mean: {w1_val.mean().item():.6f} | fc2.W Mean: {w2_val.mean().item():.6f}")

    print("\n[Step 5] MLP Integration Success!")

if __name__ == "__main__":
    test_mlp_integrated()