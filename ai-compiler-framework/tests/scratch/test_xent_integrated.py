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
from aicf_v2.layers.cross_entropy import CrossEntropyLoss

def test_xent_integrated():
    print("=== AICF CrossEntropy Autodiff Integration Test ===")
    
    # 1. 모델 정의 (Sequential 양식 사용)
    # 캡처와 컴파일 기능을 사용하기 위해 Sequential 객체로 생성합니다.
    model = aicf.Sequential([])
    
    # 설정
    batch, dims = 4, 10
    torch.manual_seed(42)
    
    # 2. 입출력 빌드
    # (B, C) logits와 (B,) targets (int32)
    logits_vid = model.input("logits", TensorSpec(shape=(batch, dims), dtype="f32"))
    targets_vid = model.input("targets", TensorSpec(shape=(batch,), dtype="i32"))
    
    # 3. CrossEntropy 레이어 추가
    loss_vid = model.add(CrossEntropyLoss(reduction="mean", name="xent"), logits_vid, targets_vid)
    
    # Sequential의 빌드 완료 상태를 위해 출력 지정
    model.output("loss", loss_vid)
    model._is_built = True # 수동 빌드 시 플래그 세팅
    
    # 4. Backward Build
    print("[Step 1] Building Backward Graph for CrossEntropy...")
    model.build_backward(loss_vid)
    
    # 5. Compile & Capture
    print("[Step 2] Compiling & Capturing Graph...")
    sample_feed = {
        "logits": torch.randn(batch, dims, device="cuda"),
        "targets": torch.randint(0, dims, (batch,), device="cuda", dtype=torch.int32),
        "grad_initial": torch.ones((1,), device="cuda")
    }
    
    # 이제 AttributeError 없이 동작합니다.
    model.compile(capture=True, sample_feed=sample_feed, mode="train")
    
    # 6. Run
    print("[Step 3] Running Captured Graph...")
    model.run(sample_feed, use_cuda_graph=True)
    
    # 7. 결과 확인 (간단한 수치 sanity check)
    gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]
    
    def get_vid_by_name(name):
        for vid, val in enumerate(model.b.values):
            if val.name == name: return vid
        return None

    d_logits_vid = get_vid_by_name("xent.d_logits")
    aicf_d_logits = gprog.slots[d_logits_vid]
    
    print(f"\nResults:")
    print(f" - d_logits Mean: {aicf_d_logits.mean().item():.6f}")
    print(f" - d_logits Finite? {torch.isfinite(aicf_d_logits).all().item()}")

    if torch.isfinite(aicf_d_logits).all():
        print("\n[SUCCESS] CrossEntropy Integration Success!")

if __name__ == "__main__":
    test_xent_integrated()