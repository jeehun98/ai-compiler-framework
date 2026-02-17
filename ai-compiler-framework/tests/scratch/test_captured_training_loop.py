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


import torch
import aicf_v2 as aicf
from aicf_v2.tensor_spec import TensorSpec
import sys

def test_captured_training():
    print("=== AICF Captured Training Loop Integration Test ===")
    
    # 1. 모델 정의 (Simple Adam Optimizer Pipeline)
    # 가중치 w를 g(gradient)를 이용해 Adam 방식으로 업데이트하는 그래프
    model = aicf.Sequential([
        # 실제로는 여러 레이어가 있겠지만, 핵심 상태 변화 확인을 위해 AdamStep 위주 구성
    ])
    
    # 수동으로 하위 레벨 그래프 구성 (Adam 연산 포함)
    shape = (1024, 1024)
    w_vid = model.param("w", TensorSpec(shape=shape, dtype="f32"))
    g_vid = model.input("g", TensorSpec(shape=shape, dtype="f32"))
    m_vid = model.state("m", TensorSpec(shape=shape, dtype="f32"))
    v_vid = model.state("v", TensorSpec(shape=shape, dtype="f32"))
    step_vid = model.state("step", TensorSpec(shape=(1,), dtype="i32"))

    # 연산 추가
    bc1, bc2 = model.add(aicf.BiasCorr(name="bc", beta1=0.9, beta2=0.999), step_vid)
    # AdamStep은 w, m, v를 In-place로 업데이트함
    p2, m2, v2 = model.add(
        aicf.AdamStep(name="adam", lr=1e-3), 
        w_vid, g_vid, m_vid, v_vid, bc1, bc2
    )
    model.add(aicf.StepInc(name="inc"), step_vid)

    # 출력 등록
    model.output("w_out", w_vid)
    model.output("step_out", step_vid)
    model._is_built = True

    # 2. 컴파일 및 사전 캡처 (Capture Phase)
    # 실제 학습을 시작하기 전, 샘플 데이터를 넣어 GPU 주소를 고정시킵니다.
    print("\n[Step 1] Compiling and Capturing Graph...")
    
    # 초기 파라미터 및 상태 설정
    init_w = torch.ones(shape, device="cuda", dtype=torch.float32)
    init_g = torch.zeros(shape, device="cuda", dtype=torch.float32)
    init_m = torch.zeros(shape, device="cuda", dtype=torch.float32)
    init_v = torch.zeros(shape, device="cuda", dtype=torch.float32)
    init_step = torch.zeros((1,), device="cuda", dtype=torch.int32)

    sample_feed = {
        "w": init_w, "g": init_g, "m": init_m, "v": init_v, "step": init_step
    }
    
    # 2. 컴파일 및 사전 캡처
    model.compile(capture=True, sample_feed=sample_feed, mode="train")

    # [수정] 캡처된 프로그램 내부의 실제 버퍼(ext_bufs)를 리셋
    # gprog는 executor 내부 캐시에 들어있으므로 이를 꺼내옵니다.
    gprog = model.executor._graph_cache[list(model.executor._graph_cache.keys())[0]]

    with torch.no_grad():
        # 캡처된 내부 버퍼들을 0으로 초기화 (가중치는 1로)
        for vid, buf in gprog.ext_bufs.items():
            name = model.b.values[vid].name
            if name == "step":
                buf.zero_()
                print(f"[Debug] Internal step buffer reset to {buf.item()}")
            elif name == "w":
                buf.fill_(1.0)
            elif name in ["m", "v"]:
                buf.zero_()

    print("\n[Step 2] Starting Training Loop (Replay Only)...")
        
    current_w = init_w.clone()
    
    for epoch in range(1, 4):
        # 매 스텝 새로운 Gradient 생성
        grad = torch.full(shape, fill_value=float(epoch), device="cuda", dtype=torch.float32)
        
        # [핵심] model.run은 이제 내부적으로 캡처된 그래프를 Replay만 함
        # 외부에서 "w"를 다시 넘겨주더라도, 캡처된 주소가 우선시됨 (State 보존)
        results = model.run({"g": grad}, use_cuda_graph=True, mode="train")
        
        updated_w = results["w_out"]
        updated_step = results["step_out"].item()

        # 검증: 가중치가 이전과 달라졌는지(학습이 되었는지) 확인
        diff = torch.abs(current_w - updated_w).max().item()
        print(f"Epoch {epoch}: Step={updated_step}, Max W Diff={diff:.6f}")
        
        assert diff > 0, "Weight should be updated by AdamStep"
        assert updated_step == epoch, f"Step count mismatch: {updated_step} != {epoch}"
        
        current_w = updated_w.clone()

    print("\n[Step 3] Verification Success!")
    print("- Graph Replay: OK")
    print("- In-place State Persistence: OK")
    print("- Pre-capture matching: OK")

if __name__ == "__main__":
    test_captured_training()