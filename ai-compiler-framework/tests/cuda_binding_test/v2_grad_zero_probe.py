from __future__ import annotations
import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
EX_PY = ROOT / "examples" / "python"
BUILD_PY = ROOT / "build" / "python"
for p in (EX_PY, BUILD_PY):
    sp = str(p)
    if sp not in sys.path:
        sys.path.insert(0, sp)
import _C

def maxabs_delta(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())

def measure_bandwidth(func, *args, numel, dtype_size, rep=100, warmup=10):
    """
    GradZero Operation:
    - Write Y (Zeros)
    => Total 1 memory access per element.
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
    
    # GB/s = (Numel * Size * 1) / Time  <-- Write Only
    total_bytes = numel * dtype_size
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return avg_ms, gbps

def run_test(shape, dtype, name, do_bench=True):
    device = torch.device("cuda:0")
    is_fp16 = (dtype == torch.float16)
    dtype_size = 2 if is_fp16 else 4
    
    # In-place simulation (GradZero is usually in-place on gradients)
    x = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    
    # Kernel Wrapper
    def _run():
        _C.op_call(int(_C.OpKind.GradZero), [x], [x], 0, b"", 0)

    # Correctness
    _run()
    d = maxabs_delta(x, torch.zeros_like(x))
    
    msg = f"[{name:<5}] Shape={str(tuple(shape)):<20} | Diff={d:.1e}"

    # Benchmark
    if do_bench:
        numel = x.numel()
        ms, gbps = measure_bandwidth(_run, numel=numel, dtype_size=dtype_size)
        msg += f" | {ms:.3f} ms | {gbps:.2f} GB/s (Write-Only)"

    print(msg)
    return d

def main():
    torch.manual_seed(0)
    print(f"GradZero Enum Value = {int(_C.OpKind.GradZero)}")
    print("-" * 90)

    # 1. Functional Checks
    run_test((1024,), torch.float32, "F32", do_bench=False)
    print("-" * 90)

    # 2. Large Benchmark (16M elements)
    # 16M * 4B = 64MB Write
    large_shape = (4096, 4096)
    
    # Note: cudaMemset speed should be roughly same for F32/F16 (byte filling)
    # GB/s calculation will normalize it.
    run_test(large_shape, torch.float32, "F32", do_bench=True)
    run_test(large_shape, torch.float16, "F16", do_bench=True)
    
    print("-" * 90)

if __name__ == "__main__":
    main()