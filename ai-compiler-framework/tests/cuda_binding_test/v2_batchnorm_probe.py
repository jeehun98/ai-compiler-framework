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

# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def schema_4cc(name: str) -> int:
    assert len(name) == 4
    b = name.encode("ascii")
    return b[0] | (b[1] << 8) | (b[2] << 16) | (b[3] << 24)

SCHEMA_BNEP = schema_4cc("BNEP")

def make_attr_bnep(eps: float, use_running_stats: bool) -> tuple[int, bytes]:
    flags = 1 if use_running_stats else 0
    return SCHEMA_BNEP, struct.pack("<fI", float(eps), int(flags))

def max_abs(a, b):
    return (a - b).abs().max().item()

def measure_bandwidth(func, *args, N, C, H, W, mode="fwd_train", rep=50, warmup=10):
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
    
    # FP16 = 2 bytes per element
    total_elements = N * C * H * W
    tensor_size = total_elements * 2 
    
    if mode == "fwd_train": factor = 3  # Read X, Read X(again for norm), Write Y
    elif mode == "fwd_infer": factor = 2 # Read X, Write Y
    elif mode == "bwd": factor = 5       # Read X, dY (sum), Read X, dY (dx), Write dX
    else: factor = 0

    total_bytes = tensor_size * factor
    gbps = total_bytes / (avg_ms / 1000.0) / 1e9
    return avg_ms, gbps

# -----------------------------------------------------------------------------
# Runners (Verbose flag added)
# -----------------------------------------------------------------------------

@torch.no_grad()
def run_fwd_training(N=8, C=16, H=8, W=8, affine=True, eps=1e-5, do_bench=False):
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    save_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
    save_rstd = torch.zeros(C, device="cuda", dtype=torch.float32)

    if affine:
        gamma = torch.randn(C, device="cuda", dtype=torch.float16)
        beta  = torch.randn(C, device="cuda", dtype=torch.float16)
        inputs = [x, gamma, beta]
    else:
        inputs = [x]

    outputs = [y, save_mean, save_rstd]
    schema_id, attrs = make_attr_bnep(eps, use_running_stats=False)

    def _run():
        _C.op_call(_C.OpKind.BatchNormFwd, inputs, outputs, schema_id, attrs)

    _run()

    # Ref
    if affine:
        y_ref = torch.nn.functional.batch_norm(x, None, None, weight=gamma, bias=beta, training=True, momentum=0.0, eps=eps)
    else:
        y_ref = torch.nn.functional.batch_norm(x, None, None, weight=None, bias=None, training=True, momentum=0.0, eps=eps)

    d = max_abs(y.float(), y_ref.float())
    
    if do_bench:
        ms, gbps = measure_bandwidth(_run, N=N, C=C, H=H, W=W, mode="fwd_train")
        tag = "Affine" if affine else "NoAff"
        print(f"[FWD Train {tag:<6}] {N}x{C}x{H}x{W:<10} | Diff={d:.1e} | {ms:.3f} ms | {gbps:.2f} GB/s")
    
    return d

@torch.no_grad()
def run_fwd_infer(N=8, C=16, H=8, W=8, affine=True, eps=1e-5, do_bench=False):
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    y = torch.empty_like(x)
    running_mean = torch.randn(C, device="cuda", dtype=torch.float32)
    running_var  = torch.rand(C, device="cuda", dtype=torch.float32) + 0.5

    if affine:
        gamma = torch.randn(C, device="cuda", dtype=torch.float16)
        beta  = torch.randn(C, device="cuda", dtype=torch.float16)
        inputs = [x, gamma, beta, running_mean, running_var]
    else:
        inputs = [x, running_mean, running_var]

    outputs = [y]
    schema_id, attrs = make_attr_bnep(eps, use_running_stats=True)

    def _run():
        _C.op_call(_C.OpKind.BatchNormFwd, inputs, outputs, schema_id, attrs)

    _run()

    y_ref = torch.nn.functional.batch_norm(
        x, running_mean=running_mean, running_var=running_var,
        weight=(gamma if affine else None), bias=(beta if affine else None),
        training=False, momentum=0.0, eps=eps
    )

    d = max_abs(y.float(), y_ref.float())
    
    if do_bench:
        ms, gbps = measure_bandwidth(_run, N=N, C=C, H=H, W=W, mode="fwd_infer")
        tag = "Affine" if affine else "NoAff"
        print(f"[FWD Infer {tag:<6}] {N}x{C}x{H}x{W:<10} | Diff={d:.1e} | {ms:.3f} ms | {gbps:.2f} GB/s")
    
    return d

def run_bwd_training(N=8, C=16, H=8, W=8, eps=1e-5, do_bench=False):
    # Inputs
    x = torch.randn(N, C, H, W, device="cuda", dtype=torch.float16)
    gamma = torch.randn(C, device="cuda", dtype=torch.float16)
    beta  = torch.randn(C, device="cuda", dtype=torch.float16)
    dy = torch.randn_like(x)

    # Get saved stats
    y_dummy = torch.empty_like(x)
    save_mean = torch.zeros(C, device="cuda", dtype=torch.float32)
    save_rstd = torch.zeros(C, device="cuda", dtype=torch.float32)
    schema_id, attrs = make_attr_bnep(eps, use_running_stats=False)
    _C.op_call(_C.OpKind.BatchNormFwd, [x, gamma, beta], [y_dummy, save_mean, save_rstd], schema_id, attrs)

    dx = torch.empty_like(x)
    dgamma = torch.zeros(C, device="cuda", dtype=torch.float32)
    dbeta  = torch.zeros(C, device="cuda", dtype=torch.float32)

    def _run():
        _C.op_call(_C.OpKind.BatchNormBwd, [x, dy, gamma, save_mean, save_rstd], [dx, dgamma, dbeta])

    _run()

    # PyTorch Ref (Float)
    x2 = x.detach().clone().float().requires_grad_(True)
    g2 = gamma.detach().clone().float().requires_grad_(True)
    b2 = beta.detach().clone().float().requires_grad_(True)
    dy_f = dy.float()

    y_ref = torch.nn.functional.batch_norm(x2, None, None, weight=g2, bias=b2, training=True, momentum=0.0, eps=eps)
    y_ref.backward(dy_f)

    err = max(max_abs(dx.float(), x2.grad), max_abs(dgamma, g2.grad), max_abs(dbeta, b2.grad))
    
    if do_bench:
        ms, gbps = measure_bandwidth(_run, N=N, C=C, H=H, W=W, mode="bwd")
        print(f"[BWD Train Affine] {N}x{C}x{H}x{W:<10} | Diff={err:.1e} | {ms:.3f} ms | {gbps:.2f} GB/s")
    
    return err

def main():
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = False
    
    print(f"BatchNorm Enum: Fwd={int(_C.OpKind.BatchNormFwd)}, Bwd={int(_C.OpKind.BatchNormBwd)}")
    print("-" * 90)

    # 1. Silent Correctness Checks
    print("Running Correctness Checks (Small sizes)... ", end="", flush=True)
    shapes = [(8,16,8,8), (16,32,16,16)]
    for (N,C,H,W) in shapes:
        err = 0.0
        err = max(err, run_fwd_training(N,C,H,W, affine=True))
        err = max(err, run_fwd_infer(N,C,H,W, affine=True))
        err = max(err, run_bwd_training(N,C,H,W))
        if err > 0.1: # Threshold check
            print(f"\n[FAIL] Size {N}x{C}x{H}x{W} Error: {err}")
            return
    print("PASS")
    print("-" * 90)

    # 2. Benchmark (Only Large)
    # N*C*H*W = 32*64*128*128 = ~33M elements
    large_cfg = (32, 64, 128, 128)
    
    print("Running Benchmarks (Large: 32x64x128x128)...")
    run_fwd_training(*large_cfg, affine=True, do_bench=True)
    run_fwd_infer(*large_cfg, affine=True, do_bench=True)
    run_bwd_training(*large_cfg, do_bench=True)

    print("-" * 90)

if __name__ == "__main__":
    main()