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
from aicf_v2.layers.softmax import Softmax

def test_softmax_autodiff():
    print("=== AICF Softmax Autodiff Numerical Test ===")
    
    # 설정
    torch.manual_seed(42)
    batch, dims = 2, 4
    x_data = torch.randn(batch, dims, device="cuda")
    y_true_data = torch.tensor([[0, 0, 1, 0], [1, 0, 0, 0]], dtype=torch.float32, device="cuda")

    # --- [1] PyTorch Reference ---
    pt_x = x_data.clone().detach().requires_grad_(True)
    pt_y = torch.softmax(pt_x, dim=-1)
    pt_loss = torch.mean((pt_y - y_true_data)**2)
    pt_loss.backward()
    pt_grad_x = pt_x.grad

    # --- [2] AICF Model ---
    model = aicf.Sequential([
        Softmax(axis=-1, name="sm1")
    ])
    
    # Build
    x_spec = TensorSpec(shape=(batch, dims), dtype="f32")
    y_pred_vid = model.build(x_spec, input_name="x")
    
    # Loss 추가
    y_true_vid = model.input("y_true", TensorSpec(shape=(batch, dims), dtype="f32"))
    loss_vid = model.add(MSELoss(reduction="mean", name="loss"), y_pred_vid, y_true_vid)
    
    # Backward Build
    print("[Step 1] Building Backward Graph for Softmax...")
    model.build_backward(loss_vid)
    
    # Compile & Capture
    sample_feed = {
        "x": x_data,
        "y_true": y_true_data,
        "grad_initial": torch.ones((1,), device="cuda")
    }
    model.compile(capture=True, sample_feed=sample_feed, mode="train")
    
    # Run
    model.run(sample_feed, use_cuda_graph=True)
    
    # --- [3] 결과 비교 ---
    gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]
    
    # x의 그라디언트 Vid 찾기
    def get_vid_by_name(name):
        for vid, val in enumerate(model.b.values):
            if val.name == name: return vid
        return None

    # Softmax 입력(x)에 대한 grad_x Vid 추출
    grad_x_vid = get_vid_by_name("sm1.dx")
    aicf_grad_x = gprog.slots[grad_x_vid]

    # 오차 확인
    diff = torch.abs(aicf_grad_x - pt_grad_x).max().item()
    
    print(f"\nResults:")
    print(f" - PyTorch Grad X Mean: {pt_grad_x.mean().item():.6f}")
    print(f" - AICF    Grad X Mean: {aicf_grad_x.mean().item():.6f}")
    print(f" - Max Absolute Diff: {diff:.2e}")

    if diff < 1e-5:
        print("\n[SUCCESS] Softmax Autodiff matches PyTorch!")
    else:
        print("\n[FAIL] Numerical mismatch in Softmax backward.")

if __name__ == "__main__":
    test_softmax_autodiff()