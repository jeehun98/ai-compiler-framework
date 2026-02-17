from __future__ import annotations
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 1) 경로 설정
p = Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
sys.path.insert(0, str(root / "python" / "aicf_v2" / "src"))

import aicf_v2 as aicf
from aicf_v2.layers.relu import ReLU
from aicf_v2.layers.softmax import Softmax
from aicf_v2.layers.cross_entropy import CrossEntropyLoss
from aicf_v2.optimizers.adam import Adam

def train_mnist():
    print("🚀 Starting AICF MNIST Training...")
    device = "cuda"
    batch_size = 64
    epochs = 5
    lr = 0.001

    # 2. 데이터 로더 준비 (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 3. AICF 모델 정의 (784 -> 128 -> 10)
    model = aicf.Sequential([
        aicf.Linear(in_features=784, out_features=128, name="fc1"),
        aicf.ReLU(name="relu1"),
        aicf.Linear(in_features=128, out_features=10, name="fc2"),
        aicf.Softmax(axis=-1, name="prob")
    ])

    # 4. 그래프 빌드 (Forward & Backward)
    x_spec = aicf.TensorSpec(shape=(batch_size, 784), dtype="f32")
    y_pred_vid = model.build(x_spec, input_name="x")
    
    y_true_vid = model.input("y_true", aicf.TensorSpec(shape=(batch_size,), dtype="i32"))
    loss_vid = model.add(CrossEntropyLoss(reduction="mean", name="loss"), y_pred_vid, y_true_vid)
    
    print("[AICF] Building Backward Graph...")
    model.build_backward(loss_vid)
    
    print("[AICF] Adding Adam Optimizer...")
    optimizer = Adam(model, lr=lr)
    optimizer.step()

    # 5. 첫 번째 배치로 그래프 캡처 (Capture)
    print("[AICF] Capturing CUDA Graph...")
    first_images, first_labels = next(iter(train_loader))
    first_images = first_images.view(batch_size, -1).to(device)
    first_labels = first_labels.to(device).to(torch.int32)

    sample_feed = {
        "x": first_images,
        "y_true": first_labels,
        "grad_initial": torch.ones((1,), device=device),
        "adam.bc1": torch.tensor([0.9], device="cuda"), # 초기 t=1 보정치
        "adam.bc2": torch.tensor([0.999], device="cuda")
    }
    model.compile(capture=True, sample_feed=sample_feed, mode="train")

    # 6. 학습 루프
    print("\n" + "="*50)
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # 데이터 전처리
            data = data.view(batch_size, -1).to(device)
            target = target.to(device).to(torch.int32)

            # [수정] feed에 grad_initial을 포함시켜야 합니다.
            # 또한 Adam 옵티마이저를 사용 중이라면 bc1, bc2도 매번 필요할 수 있습니다.
            feed = {
                "x": data, 
                "y_true": target,
                "grad_initial": torch.ones((1,), device=device),
                "adam.bc1": torch.tensor([0.9], device=device), # 일단 고정값으로 테스트
                "adam.bc2": torch.tensor([0.999], device=device)
            }
            # 그래프 실행
            model.run(feed, use_cuda_graph=True)

            # 결과 모니터링 (Softmax 출력 가져오기)
            gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]
            prob_vid = [v for v in enumerate(model.b.values) if v[1].name == "prob.out"][0][0]
            probs = gprog.slots[prob_vid]
            
            # 정확도 계산
            pred = probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:3d}/{len(train_loader)} | Running...")

        acc = 100. * correct / (len(train_loader) * batch_size)
        print(f"✅ Epoch {epoch} Done! Accuracy: {acc:.2f}%")
        print("="*50)

if __name__ == "__main__":
    train_mnist()