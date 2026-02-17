from __future__ import annotations
import torch
import sys
from pathlib import Path

# 경로 설정
p = Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
sys.path.insert(0, str(root / "python" / "aicf_v2" / "src"))

import aicf_v2 as aicf
from aicf_v2.tensor_spec import TensorSpec
from aicf_v2.emitters.cuda.sgd_step import sgd_step
from aicf_v2.layers.mse import MSELoss # 신규 추가

def test_autodiff_integrated():
    print("=== AICF Integrated Autodiff & Training Test ===")
    
    # 1. 모델 정의
    linear_layer = aicf.Linear(in_features=4, out_features=2, name="fc1", bias=True)
    model = aicf.Sequential([linear_layer])
    
    # 2. Forward Build
    x_spec = TensorSpec(shape=(1, 4), dtype="f32")
    y_pred_vid = model.build(x_spec, input_name="x")
    
    # 3. Loss Build (model.op 대신 model.add 사용)
    y_true_vid = model.input("y_true", TensorSpec(shape=(1, 2), dtype="f32"))
    loss_vid = model.add(MSELoss(reduction="mean", name="loss"), y_pred_vid, y_true_vid)
    
    # 4. Backward Build
    print("\n[Step 1] Building Backward Graph...")
    model.build_backward(loss_vid)
    
    # 5. Optimizer Build
    print("[Step 2] Adding Optimizer Ops (SGD)...")
    lr = 0.01
    for p_vid, g_vid in model.parameter_grads.items():
        sgd_step(model.b, model.ctx, P=p_vid, G=g_vid, outP=p_vid, lr=lr, 
                 name=f"update_{model.b.values[p_vid].name}")

    # 6. Compile & Capture
    print("[Step 3] Compiling & Capturing Integrated Graph...")
    torch.manual_seed(42)
    sample_feed = {
        "x": torch.randn(1, 4, device="cuda"), 
        "y_true": torch.ones(1, 2, device="cuda") * 0.5,
        "grad_initial": torch.ones((1,), device="cuda") 
    }
    
    # Capture (이제 모든 Op가 kind_id를 가지므로 성공합니다)
    model.compile(capture=True, sample_feed=sample_feed, mode="train")
    
    # 7. Training Loop
    print("\n[Step 4] Training Loop Starting (Replay)...")
    gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]
    w_vid = model._tape[0]['params'][0] # fc1.W
    
    for i in range(10):
        model.run(sample_feed, use_cuda_graph=True)
        w_val = gprog.slots[w_vid]
        print(f"Iter {i}: Weight Mean = {w_val.mean().item():.6f}")

    print("\n[Step 5] Integration Success!")

if __name__ == "__main__":
    test_autodiff_integrated()