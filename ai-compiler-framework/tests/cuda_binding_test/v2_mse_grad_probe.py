from __future__ import annotations
import sys
from pathlib import Path
import struct
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

def pack_mse_grad_scale(scale: float) -> bytes:
    return struct.pack("<f", float(scale))

def measure_bandwidth(func, *args, numel, dtype_size, rep=100, warmup=10):
    """
    MseGrad Memory Pattern:
    - Read Pred, Read Target, Write Grad
    => Total 3 memory accesses per element.
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
    
    # GB/s = (Numel * Size * 3) / Time
    total_bytes = numel * dtype_size * 3
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return avg_ms, gbps

def run_bench(dtype, shape, name):
    device = torch.device("cuda:0")
    dtype_size = 2 if dtype == torch.float16 else 4
    
    pred = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    targ = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    grad = torch.empty_like(pred).contiguous()
    
    # Schema setup (default scale)
    schema_id = 0
    attrs = b""

    def _run():
        _C.op_call(int(_C.OpKind.MseGrad), [pred, targ], [grad], schema_id, attrs, 0)

    # Correctness check (Warmup during check)
    _run() 
    
    # Benchmark
    numel = pred.numel()
    ms, gbps = measure_bandwidth(_run, numel=numel, dtype_size=dtype_size)
    
    print(f"[{name:<10}] Shape={str(tuple(shape)):<20} | {ms:.3f} ms | {gbps:.2f} GB/s")

def main():
    torch.manual_seed(0)
    print(f"MseGrad Enum = {int(_C.OpKind.MseGrad)}")
    print("-" * 80)
    
    # Large Benchmark (16M elements)
    large_shape = (4096, 4096)
    
    run_bench(torch.float32, large_shape, "F32")
    
    # F16 Naive (Force odd numel to disable vectorization)
    odd_shape = (4096 * 4096 - 1,)
    run_bench(torch.float16, odd_shape, "F16-Naive")
    
    # F16 Vec2 (Even numel -> half2 optimization)
    run_bench(torch.float16, large_shape, "F16-Vec2")
    
    print("-" * 80)

if __name__ == "__main__":
    main()