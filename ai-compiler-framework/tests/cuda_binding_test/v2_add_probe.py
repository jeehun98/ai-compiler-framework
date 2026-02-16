from __future__ import annotations
import sys
from pathlib import Path
import torch
import struct

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)

import _C


def measure_bandwidth(func, *args, numel, dtype_size, rep=100, warmup=10):
    """
    Measures execution time and calculates Effective Bandwidth (GB/s).
    Add Op: 2 Reads + 1 Write = 3 memory accesses per element.
    """
    # 1. Warmup
    for _ in range(warmup):
        func(*args)
    torch.cuda.synchronize()

    # 2. Measure
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    start_event.record()
    for _ in range(rep):
        func(*args)
    end_event.record()
    
    torch.cuda.synchronize()
    
    avg_ms = start_event.elapsed_time(end_event) / rep
    
    # GB/s = (Elements * ItemSize * 3) / (Seconds * 1e9)
    total_bytes = numel * dtype_size * 3
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    
    return avg_ms, gbps

def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

def run_test(shape, dtype, name, do_bench=True):
    is_fp16 = (dtype == torch.float16)
    dtype_size = 2 if is_fp16 else 4
    
    a = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    b = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    o = torch.empty_like(a).contiguous()

    # Ref
    if is_fp16:
        ref = (a.float() + b.float()).half()
    else:
        ref = a + b

    # Kernel Wrapper
    def _run():
        _C.op_call(
            int(_C.OpKind.EltwiseAdd),
            [a, b],
            [o],
            0,
            b"",
            0,
        )

    # Correctness
    _run()
    d = maxabs_delta(o, ref)
    
    msg = f"[{name:<8}] Shape={str(tuple(shape)):<20} | Diff={d:.1e}"
    
    # Benchmark
    if do_bench:
        numel = a.numel()
        ms, gbps = measure_bandwidth(_run, numel=numel, dtype_size=dtype_size)
        msg += f" | Time={ms:.3f} ms | BW={gbps:.2f} GB/s"
        
    print(msg)
    return d

def main():
    torch.manual_seed(0)
    print(f"Add enum value = {int(_C.OpKind.EltwiseAdd)}")
    print("-" * 90)
    
    # 1. Large size to saturate bandwidth (16M elements)
    # 16M * 4B = 64MB per tensor
    large_shape = (4096, 4096) 
    
    # F32 Naive
    run_test(large_shape, torch.float32, "F32")
    
    print("-" * 90)
    
    # F16 Naive (Odd size -> Vec2 not applicable)
    # Force odd number of elements to hit the 'naive' kernel path
    odd_shape = (4096 * 4096 - 1,)
    run_test(odd_shape, torch.float16, "F16-Naive")

    # F16 Vec2 (Even size, Aligned -> Vec2 Fastpath)
    run_test(large_shape, torch.float16, "F16-Vec2")

    print("=" * 90)

    # Negative Cases
    print("\n[Negative Tests]")
    # NEG1: shape mismatch
    try:
        a = torch.randn(10, device="cuda")
        b = torch.randn(11, device="cuda")
        o = torch.empty_like(a)
        _C.op_call(int(_C.OpKind.EltwiseAdd), [a, b], [o], 0, b"", 0)
    except RuntimeError as e:
        print(f"[Shape Mismatch] OK: {str(e).splitlines()[0]}")

if __name__ == "__main__":
    main()