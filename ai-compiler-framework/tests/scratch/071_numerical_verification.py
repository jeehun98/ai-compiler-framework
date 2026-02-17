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
from aicf_v2.layers import Linear, Softmax
from aicf_v2.runtime.cuda_exec import CudaExecutor

def test_numerical():
    # 1. 모델 세팅 (Sequential)
    model = aicf.Sequential([
        Linear(128, 10, name="fc1"),
        Softmax(axis=-1, name="prob")
    ], dtype="f32")
    
    model.build(aicf.TensorSpec(shape=(4, 128)))
    exe = CudaExecutor()
    compiled = model.compile()

    # 2. 데이터 준비
    torch.manual_seed(42)
    x = torch.randn(4, 128, device="cuda")
    w = torch.randn(10, 128, device="cuda")
    b = torch.randn(10, device="cuda")
    
    # AICF 피드 데이터 구성
    feed = {
        "x": x,
        "fc1.W": w,
        "fc1.b": b
    }

    # 3. AICF 실행
    out_aicf = exe.run(model, feed, use_cuda_graph=False)
    y_aicf = out_aicf["output"]

    # 4. Torch 기대값 계산 (검증용)
    with torch.no_grad():
        y_torch = torch.nn.functional.linear(x, w, b)
        y_torch = torch.nn.functional.softmax(y_torch, dim=-1)

    # 5. 비교
    diff = (y_aicf - y_torch).abs().max().item()
    print(f"\nMax Difference: {diff:.8e}")
    
    if diff < 1e-5:
        print("✅ Numerical Verification SUCCESS!")
    else:
        print("❌ Numerical Verification FAILED!")

if __name__ == "__main__":
    test_numerical()