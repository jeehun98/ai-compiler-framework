from __future__ import annotations
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
import math
import struct
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

def initialize_weights_kaiming(model):
    """모델 파라미터를 Kaiming Normal 방식으로 초기화합니다."""
    print("\n[DEBUG] Initializing weights...")
    for name, tensor in model.parameters.items():
        if "bias" in name or ".b" in name:
            tensor.zero_()
        else:
            fan_in = tensor.shape[1] if len(tensor.shape) > 1 else tensor.shape[0]
            std = math.sqrt(2.0 / fan_in)
            torch.nn.init.normal_(tensor, mean=0.0, std=std)
        print(f" -> {name}: shape={list(tensor.shape)}, mean={tensor.mean().item():.6f}")

def train_mnist():
    device = "cuda"
    batch_size = 64
    epochs = 5
    lr = 0.001

    # 1. 데이터 로더 (drop_last=True로 설정하여 배수 불일치 차단)
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
    ])

    # 3. 그래프 빌드
    x_spec = aicf.TensorSpec(shape=(batch_size, 784), dtype="f32", device="cuda")
    y_pred_vid = model.build(x_spec, input_name="x")
    # 타겟은 명확하게 1D [batch_size] i32로 설정
    y_true_vid = model.input("y_true", aicf.TensorSpec(shape=(batch_size,), dtype="i32", device="cuda"))
    
    loss_vid = model.add(
        aicf.CrossEntropyLoss(reduction="mean", name="loss"), 
        y_pred_vid, y_true_vid,
        out_spec=aicf.TensorSpec(shape=(1,), dtype="f32", device="cuda")
    )
    model.b.outputs["prob"] = y_pred_vid
    model.b.outputs["loss"] = loss_vid

    # 4. 초기화 및 빌드
    initialize_weights_kaiming(model)
    model.build_backward(loss_vid)
    optimizer = Adam(model, lr=lr)
    optimizer.step()

    # 5. 고정 버퍼 설정 (DType 및 메모리 연속성 강제)
    static_input = torch.zeros((batch_size, 784), device=device, dtype=torch.float32).contiguous()
    static_target = torch.zeros((batch_size,), device=device, dtype=torch.int32).contiguous()
    static_grad_init = torch.ones((1,), device=device, dtype=torch.float32).contiguous()
    
    curr_bc1, curr_bc2 = 0.9, 0.999
    model.compile(capture=False, mode="train") 

    print("\n" + "="*65)
    print(f"🚀 Starting Training (CUDA Graph: False)")
    print("="*65)
    
    # 6. 학습 루프 설정
    step = 0 
    beta1 = 0.9
    beta2 = 0.999

    for epoch in range(epochs):
        # ✅ 여기서 변수 초기화 (에러 해결 포인트)
        total_loss = 0.0
        correct = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            step += 1
            
            # 메모리 복사 및 전처리
            static_input.copy_(data.view(batch_size, -1))
            static_target.copy_(target.to(torch.int32))

            # ✅ Adam Bias Correction 수식 수정 (1 - beta^t)
            curr_bc1 = 1.0 - (beta1 ** step)
            curr_bc2 = 1.0 - (beta2 ** step)

            feed = {
                "x": static_input, 
                "y_true": static_target,
                "grad_initial": static_grad_init,
                "adam.bc1": torch.tensor([curr_bc1], device=device, dtype=torch.float32),
                "adam.bc2": torch.tensor([curr_bc2], device=device, dtype=torch.float32)
            }
            
            print(model, "내용 확인")

            # 모델 실행
            outputs = model.run(feed, use_cuda_graph=False, mode="train")
            
            # 손실값 및 정확도 누적
            loss_val = outputs["loss"].item()
            total_loss += loss_val  # 이제 정상 작동합니다
            
            pred = outputs["prob"].argmax(dim=1)
            correct += pred.eq(static_target).sum().item()

            if batch_idx % 100 == 0:
                print(f"Batch {batch_idx:4d}/{len(train_loader)} | Loss: {loss_val:.6f}")

        avg_acc = 100. * correct / (len(train_loader) * batch_size)
        print(f"\n✅ Epoch {epoch} Done! | Avg Loss: {total_loss/len(train_loader):.4f} | Acc: {avg_acc:.2f}%")
        print("="*65)

if __name__ == "__main__":
    train_mnist()