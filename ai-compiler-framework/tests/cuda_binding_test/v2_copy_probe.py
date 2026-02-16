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
    Copy Operation:
    - Read X
    - Write Y
    => Total 2 memory accesses per element.
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
    
    # GB/s = (Numel * Size * 2) / Time
    total_bytes = numel * dtype_size * 2
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return avg_ms, gbps

def run_test(shape, dtype, name, do_bench=True):
    device = torch.device("cuda:0")
    is_fp16 = (dtype == torch.float16)
    dtype_size = 2 if is_fp16 else 4
    
    x = torch.randn(*shape, device=device, dtype=dtype).contiguous()
    y = torch.empty_like(x).contiguous()

    # Kernel Wrapper
    def _run():
        _C.op_call(int(_C.OpKind.Copy), [x], [y], 0, b"", 0)

    # Correctness
    _run()
    if is_fp16:
        d = maxabs_delta(y.float(), x.float())
    else:
        d = maxabs_delta(y, x)
    
    msg = f"[{name:<5}] Shape={str(tuple(shape)):<20} | Diff={d:.1e}"

    # Benchmark
    if do_bench:
        numel = x.numel()
        ms, gbps = measure_bandwidth(_run, numel=numel, dtype_size=dtype_size)
        msg += f" | {ms:.3f} ms | {gbps:.2f} GB/s"

    print(msg)
    return d

def main():
    torch.manual_seed(0)
    print(f"Copy Enum Value = {int(_C.OpKind.Copy)}")
    print("-" * 90)

    # 1. Functional Checks (Small)
    run_test((1024,), torch.float32, "F32", do_bench=False)
    run_test((1024,), torch.float16, "F16", do_bench=False)
    print("-" * 90)

    # 2. Large Benchmark (16M elements)
    # 16M * 4B = 64MB (Read) + 64MB (Write) = 128MB
    large_shape = (4096, 4096)
    
    run_test(large_shape, torch.float32, "F32", do_bench=True)
    run_test(large_shape, torch.float16, "F16", do_bench=True)
    
    print("-" * 90)

    # Negative checks
    x = torch.randn(128, device="cuda", dtype=torch.float32)
    y_bad = torch.empty(128, device="cuda", dtype=torch.float16)
    try:
        _C.op_call(int(_C.OpKind.Copy), [x], [y_bad], 0, b"", 0)
    except RuntimeError as e:
        print(f"[NEG dtype] OK: {str(e).splitlines()[0]}")

if __name__ == "__main__":
    main()