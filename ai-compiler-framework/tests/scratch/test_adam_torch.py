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
from aicf_v2.optimizers.adam import Adam

def test_adam_numerical_parity():
    print("=== AICF vs PyTorch Adam Numerical Parity Test ===")
    
    # 설정
    in_dim, out_dim = 4, 2
    lr, b1, b2, eps = 0.001, 0.9, 0.999, 1e-8
    torch.manual_seed(42)

    # --- [준비] 공통 초기 가중치 및 데이터 ---
    init_W = torch.randn(out_dim, in_dim, device="cuda")
    init_b = torch.zeros(out_dim, device="cuda")
    x_data = torch.randn(1, in_dim, device="cuda")
    y_true_data = torch.ones(1, out_dim, device="cuda") * 0.5

    # --- [1] PyTorch Reference 계산 ---
    pt_W = init_W.clone().detach().requires_grad_(True)
    pt_b = init_b.clone().detach().requires_grad_(True)
    pt_opt = torch.optim.Adam([pt_W, pt_b], lr=lr, betas=(b1, b2), eps=eps)

    # PyTorch 1회 업데이트
    pred = torch.mm(x_data, pt_W.t()) + pt_b
    loss = torch.mean((pred - y_true_data)**2)
    pt_opt.zero_grad()
    loss.backward()
    pt_opt.step()

    # --- [2] AICF 계산 ---
    model = aicf.Sequential([
        aicf.Linear(in_features=in_dim, out_features=out_dim, name="fc1", bias=True)
    ])
    # 가중치 강제 동기화
    model.parameters["fc1.W"] = init_W.clone()
    model.parameters["fc1.b"] = init_b.clone()

    # 그래프 구성
    x_spec = TensorSpec(shape=(1, in_dim), dtype="f32")
    y_pred_vid = model.build(x_spec, input_name="x")
    y_true_vid = model.input("y_true", TensorSpec(shape=(1, out_dim), dtype="f32"))
    loss_vid = model.add(MSELoss(reduction="mean", name="loss"), y_pred_vid, y_true_vid)

    model.build_backward(loss_vid)
    optimizer = Adam(model, lr=lr, beta1=b1, beta2=b2, eps=eps)
    optimizer.step()

    # 캡처를 위한 피드 구성 (t=1 시점의 bc1, bc2 계산)
    sample_feed = {
        "x": x_data,
        "y_true": y_true_data,
        "grad_initial": torch.ones((1,), device="cuda"),
        "adam.bc1": torch.tensor([b1 ** 1], device="cuda"),
        "adam.bc2": torch.tensor([b2 ** 1], device="cuda")
    }

    model.compile(capture=True, sample_feed=sample_feed, mode="train")
    model.run(sample_feed, use_cuda_graph=True)

    # --- [3] 결과 비교 ---
    # AICF 결과 슬롯에서 데이터 가져오기
    gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]
    w_vid = model.b.param_vids[0] # fc1.W
    aicf_W = gprog.slots[w_vid]

    # 오차 계산
    diff = torch.abs(aicf_W - pt_W).max().item()
    
    print(f"\nResults after 1 Step:")
    print(f" - PyTorch W Mean: {pt_W.mean().item():.6f}")
    print(f" - AICF    W Mean: {aicf_W.mean().item():.6f}")
    print(f" - Max Absolute Diff: {diff:.2e}")

    if diff < 1e-6:
        print("\n[SUCCESS] AICF Adam matches PyTorch results!")
    else:
        print("\n[FAIL] Numerical mismatch detected.")

if __name__ == "__main__":
    test_adam_numerical_parity()