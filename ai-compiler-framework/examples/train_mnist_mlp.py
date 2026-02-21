from __future__ import annotations
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 1) 경로 및 라이브러리 설정
p = Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())

build_lib_path = root / "build" / "python" / "aicf_cuda"
src_path = root / "python" / "aicf_v2" / "src"

if build_lib_path.exists():
    sys.path.insert(0, str(build_lib_path))
sys.path.insert(0, str(src_path))

import aicf_v2 as aicf
from aicf_v2.optimizers.adam import Adam

# 바이너리 로드 확인
try:
    import _C
    print(f"✅ Loaded binary _C from: {_C.__file__}")
except:
    print("❌ Failed to load _C binary.")

def train_mnist():
    print("🚀 Starting AICF MNIST Training (v2 Final Mode)...")
    device = "cuda"
    batch_size = 64
    epochs = 5
    lr = 0.001

    # 1. 데이터 로더 (MNIST)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. 모델 정의
    model = aicf.Sequential([
        aicf.Linear(in_features=784, out_features=128, name="fc1"),
        aicf.ReLU(name="relu1"),
        aicf.Linear(in_features=128, out_features=10, name="fc2"),
        aicf.Softmax(axis=-1, name="prob")
    ])

    # 3. Forward 그래프 빌드
    x_spec = aicf.TensorSpec(shape=(batch_size, 784), dtype="f32", device="cuda")
    y_pred_vid = model.build(x_spec, input_name="x")
    
    y_true_vid = model.input("y_true", aicf.TensorSpec(shape=(batch_size,), dtype="i32", device="cuda"))
    
    loss_spec = aicf.TensorSpec(shape=(1,), dtype="f32", device="cuda")
    loss_vid = model.add(
        aicf.CrossEntropyLoss(reduction="mean", name="loss"), 
        y_pred_vid, 
        y_true_vid,
        out_spec=loss_spec 
    )

    # 🚀 [핵심 수정] 출력 키 명시적 등록
    # 이렇게 등록해야 model.run()이 반환하는 딕셔너리에 포함됩니다.
    model.b.outputs["prob"] = y_pred_vid
    model.b.outputs["loss"] = loss_vid

    # 4. Backward & Optimizer 설정
    print("[AICF] Building Backward Graph...")
    model.build_backward(loss_vid)
    
    print("[AICF] Adding Adam Step...")
    optimizer = Adam(model, lr=lr)
    optimizer.step()

    # 5. 컴파일 및 CUDA Graph 캡처
    print("[AICF] Capturing CUDA Graph...")
    first_images, first_labels = next(iter(train_loader))
    first_images = first_images.view(batch_size, -1).to(device).float().contiguous()
    first_labels = first_labels.to(device).to(torch.int32).contiguous()

    sample_feed = {
        "x": first_images,
        "y_true": first_labels,
        "grad_initial": torch.ones((1,), dtype=torch.float32, device=device), 
        "adam.bc1": torch.tensor([0.9], device=device),
        "adam.bc2": torch.tensor([0.999], device=device)
    }
    
    model.compile(capture=True, sample_feed=sample_feed, mode="train")

    # 6. 학습 루프
    print("\n" + "="*50)
    for epoch in range(epochs):
        correct = 0
        total_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.view(batch_size, -1).to(device).float().contiguous()
            target = target.to(device).to(torch.int32).contiguous()

            feed = {
                "x": data, 
                "y_true": target,
                "grad_initial": torch.ones((1,), dtype=torch.float32, device=device),
                "adam.bc1": torch.tensor([0.9], device=device),
                "adam.bc2": torch.tensor([0.999], device=device)
            }
            
            # 🚀 모델 실행 및 결과 획득
            outputs = model.run(feed, use_cuda_graph=True, mode="train")

            # 등록한 키 이름으로 결과 추출
            probs = outputs["prob"]
            loss_val = outputs["loss"]

            pred = probs.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total_loss += loss_val.item()

            if batch_idx % 100 == 0:
                avg_l = total_loss / (batch_idx + 1)
                print(f"Epoch {epoch} | Batch {batch_idx:3d}/{len(train_loader)} | Loss: {avg_l:.4f}")

        acc = 100. * correct / (len(train_loader) * batch_size)
        print(f"✅ Epoch {epoch} Done! Accuracy: {acc:.2f}% | Avg Loss: {total_loss/len(train_loader):.4f}")
        print("="*50)

if __name__ == "__main__":
    train_mnist()