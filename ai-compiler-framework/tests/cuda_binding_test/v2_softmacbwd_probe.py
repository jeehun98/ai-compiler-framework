from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F

# 프로젝트 루트 및 빌드 경로 설정
THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]  # 구조에 맞게 조정 (aicf-compiler-framework 기준)
BUILD_PY = ROOT / "build" / "python"
if str(BUILD_PY) not in sys.path:
    sys.path.insert(0, str(BUILD_PY))

import _C


def measure_bwd_performance(func, *args, numel, dtype_size, rep=100, warmup=10):
    """
    SoftmaxBwd 실행 시간 및 유효 대역폭(GB/s) 측정.
    SoftmaxBwd (Y,dY -> dX):
      - Read: Y 1회 + dY 1회
      - Write: dX 1회
    => 최소 3x traffic 가정 (유효 BW lower bound)
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

    total_bytes = numel * dtype_size * 3
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return avg_ms, gbps


def run_softmax_bwd_test(shape, dtype, name, do_bench=True):
    is_fp16 = (dtype == torch.float16)
    dtype_size = 2 if is_fp16 else 4

    # 입력 생성
    # bwd는 Y, dY가 입력이므로 forward로 Y를 만들어서 넣고,
    # dY는 임의로 생성
    x = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    y = F.softmax(x.float(), dim=-1).to(dtype).contiguous()
    dy = torch.randn_like(y).contiguous()
    dx = torch.empty_like(y).contiguous()

    # Reference:
    # dx = y * (dy - sum(dy*y))
    # (PyTorch autograd로 구하면 깔끔)
    x_ref = x.detach().clone().contiguous().requires_grad_(True)
    y_ref = F.softmax(x_ref.float(), dim=-1).to(dtype)
    (y_ref * dy).sum().backward()
    ref_dx = x_ref.grad.detach()

    # Kernel Wrapper: inputs=(Y, dY) outputs=(dX)
    def _run():
        _C.op_call(
            int(_C.OpKind.SoftmaxBwd),
            [y, dy],
            [dx],
            0,    # schema_id
            b"",  # attr
            0,    # stream
        )

    # 1) Correctness
    _run()
    diff = (dx.float() - ref_dx.float()).abs().max().item()
    tol = 1e-3 if is_fp16 else 1e-6
    status = "PASS" if diff < tol else "FAIL"

    msg = f"[{name:<12}] Shape={str(tuple(shape)):<20} | Diff={diff:.2e} | {status}"

    # 2) Benchmark
    if do_bench:
        numel = dx.numel()
        ms, gbps = measure_bwd_performance(_run, numel=numel, dtype_size=dtype_size)
        msg += f" | Time={ms:.3f} ms | BW={gbps:.2f} GB/s"

    print(msg)
    return diff


def main():
    torch.manual_seed(42)

    # OpKind 확인
    try:
        bwd_kind = int(_C.OpKind.SoftmaxBwd)
    except AttributeError:
        print("Error: OpKind.SoftmaxBwd not found. Please bind/register it.")
        return

    print(f"SoftmaxBwd enum value = {bwd_kind}")
    print("-" * 100)

    # 케이스 1: 작은 2D
    run_softmax_bwd_test((32, 1024), torch.float32, "BWD-F32-S")
    run_softmax_bwd_test((32, 1024), torch.float16, "BWD-F16-S")

    print("-" * 100)

    # 케이스 2: 큰 2D
    large_shape = (2048, 8192)
    run_softmax_bwd_test(large_shape, torch.float32, "BWD-F32-L")
    run_softmax_bwd_test(large_shape, torch.float16, "BWD-F16-L")

    print("-" * 100)

    # Negative Test 1: alias 금지 (dX가 Y를 alias)
    print("[Negative Test - dX aliases Y]")
    try:
        x = torch.randn(1024, device="cuda", dtype=torch.float32)
        y = F.softmax(x, dim=-1).contiguous()
        dy = torch.randn_like(y).contiguous()
        _C.op_call(int(_C.OpKind.SoftmaxBwd), [y, dy], [y], 0, b"", 0)  # output==input
        print("FAIL: alias should have been blocked.")
    except RuntimeError as e:
        print(f"OK: Caught expected error: {str(e).splitlines()[0]}")

    # Negative Test 2: wrong arity (inputs 1개)
    print("[Negative Test - wrong arity]")
    try:
        y = torch.randn(1024, device="cuda", dtype=torch.float32).contiguous()
        dx = torch.empty_like(y).contiguous()
        _C.op_call(int(_C.OpKind.SoftmaxBwd), [y], [dx], 0, b"", 0)
        print("FAIL: wrong arity should have been blocked.")
    except RuntimeError as e:
        print(f"OK: Caught expected error: {str(e).splitlines()[0]}")


if __name__ == "__main__":
    main()
