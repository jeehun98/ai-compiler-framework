from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# 프로젝트 루트 및 빌드 경로 설정
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3] # 구조에 맞게 조정 (aicf-compiler-framework 기준)
BUILD_PY = ROOT / "build" / "python"
if str(BUILD_PY) not in sys.path:
    sys.path.insert(0, str(BUILD_PY))

import _C

def measure_softmax_performance(func, *args, numel, dtype_size, rep=100, warmup=10):
    """
    Softmax 실행 시간 및 유효 대역폭(GB/s) 측정.
    Softmax: 기본적으로 Read 1회(X), Write 1회(Y)를 수행한다고 가정.
    (내부적으로 여러 번 읽더라도 메모리 인터페이스 관점의 대역폭 측정)
    """
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        func(*args)
    end_event.record()
    torch.cuda.synchronize()
    
    avg_ms = start_event.elapsed_time(end_event) / rep
    
    # GB/s = (Elements * ItemSize * 2) / (Seconds * 1e9)
    total_bytes = numel * dtype_size * 2
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    
    return avg_ms, gbps

def run_softmax_test(shape, dtype, name, do_bench=True):
    is_fp16 = (dtype == torch.float16)
    dtype_size = 2 if is_fp16 else 4
    
    # 입력 데이터 생성 (Softmax는 값이 너무 크면 정밀도 차이가 벌어질 수 있음)
    x = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    y = torch.empty_like(x).contiguous()

    # Reference: PyTorch의 Softmax (마지막 차원 기준)
    # 수치 안정성을 위해 float32에서 계산 후 캐스팅 추천
    ref = F.softmax(x.float(), dim=-1).to(dtype)

    # Kernel Wrapper
    # OpKind.Softmax가 등록되어 있다고 가정 (보통 EltwiseAdd 근처 번호)
    def _run():
        _C.op_call(
            int(_C.OpKind.Softmax), # Enum에 Softmax가 추가되어 있어야 함
            [x],       # inputs
            [y],       # outputs
            0,         # axis (현재 커널은 last_dim 고정이나 인터페이스상 유지)
            b"",       # attr
            0,         # stream
        )

    # 1. Correctness
    _run()
    diff = (y.float() - ref.float()).abs().max().item()
    
    # FP16의 경우 허용 오차를 조금 더 줌 (보통 1e-3 내외)
    tol = 1e-3 if is_fp16 else 1e-6
    status = "PASS" if diff < tol else "FAIL"
    
    msg = f"[{name:<10}] Shape={str(tuple(shape)):<20} | Diff={diff:.2e} | {status}"
    
    # 2. Benchmark
    if do_bench:
        numel = x.numel()
        ms, gbps = measure_softmax_performance(_run, numel=numel, dtype_size=dtype_size)
        msg += f" | Time={ms:.3f} ms | BW={gbps:.2f} GB/s"
        
    print(msg)
    return diff

def main():
    torch.manual_seed(42)
    # OpKind 확인 (등록된 값에 따라 수정 필요)
    try:
        softmax_kind = int(_C.OpKind.Softmax)
    except AttributeError:
        print("Error: OpKind.Softmax not found. Please register it in registry.hpp/cpp")
        return

    print(f"Softmax enum value = {softmax_kind}")
    print("-" * 100)
    
    # 테스트 케이스 1: 일반적인 2D Matrix (LLM의 Attention Head 느낌)
    # 32 heads * 1024 seq_len -> (32, 1024)
    run_softmax_test((32, 1024), torch.float32, "F32-Small")
    run_softmax_test((32, 1024), torch.float16, "F16-Small")
    
    print("-" * 100)
    
    # 테스트 케이스 2: 대규모 데이터 (Bandwidth 포화 확인)
    # (2048, 8192) 크기
    large_shape = (2048, 8192)
    run_softmax_test(large_shape, torch.float32, "F32-Large")
    run_softmax_test(large_shape, torch.float16, "F16-Large")

    print("-" * 100)

    # Negative Test: In-place 체크 (작성하신 커널에서 막아두었으므로 확인 필요)
    print("[Negative Test - In-place Alias]")
    try:
        x_alias = torch.randn(1024, device="cuda")
        _C.op_call(softmax_kind, [x_alias], [x_alias], 0, b"", 0)
        print("FAIL: In-place should have been blocked.")
    except RuntimeError as e:
        print(f"OK: Caught expected error: {str(e).splitlines()[0]}")

if __name__ == "__main__":
    main()