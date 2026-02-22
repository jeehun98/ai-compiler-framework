from __future__ import annotations
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import sys
from pathlib import Path

# 경로 설정
p = Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
build_lib_path = root / "build" / "python" / "aicf_cuda"
src_path = root / "python" / "aicf_v2" / "src"
if build_lib_path.exists(): sys.path.insert(0, str(build_lib_path))
sys.path.insert(0, str(src_path))

import aicf_v2 as aicf
from aicf_v2.optimizers.adam import Adam

def train_mnist_with_graph():
    device = "cuda"
    batch_size = 64
    epochs = 5
    lr = 0.001
    beta1, beta2 = 0.9, 0.999

    # 1. 데이터 로더
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    # 2. 모델 정의 및 그래프 빌드
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
        aicf.Linear(128, 10, name="fc2"),
    ])

    x_spec = aicf.TensorSpec(shape=(batch_size, 784), dtype="f32", device="cuda")
    y_pred_vid = model.build(x_spec, input_name="x")
    y_true_vid = model.input("y_true", aicf.TensorSpec(shape=(batch_size,), dtype="i32", device="cuda"))
    
    loss_vid = model.add(
        aicf.CrossEntropyLoss(reduction="mean", name="loss"), 
        y_pred_vid, y_true_vid
    )
    model.b.outputs["prob"] = y_pred_vid
    model.b.outputs["loss"] = loss_vid

    # 3. 초기화 및 역전파 빌드
    model.build_backward(loss_vid)
    optimizer = Adam(model, lr=lr)
    optimizer.step()

    # 4. 고정 버퍼 (CUDA Graph의 핵심: 주소 고정)
    static_input = torch.zeros((batch_size, 784), device=device, dtype=torch.float32).contiguous()
    static_target = torch.zeros((batch_size,), device=device, dtype=torch.int32).contiguous()
    static_grad_init = torch.ones((1,), device=device, dtype=torch.float32).contiguous()
    
# 5. [수정 완료] CUDA Graph 컴파일 및 캡처
    print("\n[AICF] Capturing CUDA Graph...")
    
    # 캡처용 샘플 피드 (실제 run 시 필요한 모든 키 포함)
    s_feed = {
        "x": static_input, 
        "y_true": static_target,
        "grad_initial": static_grad_init,
        "adam.bc1": torch.tensor([0.9], device=device, dtype=torch.float32),
        "adam.bc2": torch.tensor([0.999], device=device, dtype=torch.float32)
    }

    # 인자 이름을 sample_feed로 정확히 지정하여 호출
    model.compile(capture=True, sample_feed=s_feed, mode="train") 

    print("[AICF] Graph Capture Success. Starting Training Loop...")
    
    step = 0
    print("\n" + "="*65)
    print("🚀 Starting Training with CUDA Graph (High Performance)")
    print("="*65)

    for epoch in range(epochs):
        total_loss, correct = 0.0, 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            step += 1
            
            # 고정 버퍼에 데이터 복사 (원본 주소 유지)
            static_input.copy_(data.view(batch_size, -1))
            static_target.copy_(target.to(torch.int32))

            # Bias Correction 계산
            bc1, bc2 = 1.0 - (beta1 ** step), 1.0 - (beta2 ** step)

            feed = {
                "x": static_input, 
                "y_true": static_target,
                "grad_initial": static_grad_init,
                "adam.bc1": torch.tensor([bc1], device=device, dtype=torch.float32),
                "adam.bc2": torch.tensor([bc2], device=device, dtype=torch.float32)
            }
            
            # 6. 모델 실행 (use_cuda_graph=True)
            # 내부적으로 Replay를 수행하여 CPU-GPU 오버헤드를 거의 0으로 만듭니다.
            outputs = model.run(feed, use_cuda_graph=True, mode="train")
            
            loss_val = outputs["loss"].item()
            total_loss += loss_val
            pred = outputs["prob"].argmax(dim=1)
            correct += pred.eq(static_target).sum().item()

            if batch_idx % 200 == 0:
                print(f"Epoch {epoch} | Batch {batch_idx:4d}/{len(train_loader)} | Loss: {loss_val:.4f}")

        avg_acc = 100. * correct / (len(train_loader) * batch_size)
        print(f"✅ Epoch {epoch} Done! | Avg Loss: {total_loss/len(train_loader):.4f} | Acc: {avg_acc:.2f}%")

if __name__ == "__main__":
    train_mnist_with_graph()