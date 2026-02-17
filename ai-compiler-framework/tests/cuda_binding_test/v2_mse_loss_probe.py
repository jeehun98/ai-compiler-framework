from __future__ import annotations
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import struct

THIS = Path(__file__).resolve()
ROOT = THIS.parents[3]
BUILD_PY = ROOT / "build" / "python"
if str(BUILD_PY) not in sys.path:
    sys.path.insert(0, str(BUILD_PY))

import _C

def measure_loss_perf(run, rep=200, warmup=20):
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(rep):
        run()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / rep

def run_mse_loss_test(shape, dtype, name, reduction="mean", do_bench=True):
    pred = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    target = torch.randn(*shape, device="cuda", dtype=dtype).contiguous()
    out = torch.empty(1, device="cuda", dtype=torch.float32).contiguous()  # scalar f32

    ref = F.mse_loss(pred.float(), target.float(), reduction=reduction)  # scalar f32

    # Attr: schema_id=0이면 default(mean). sum 쓰고 싶으면 MSEL + int32(1)
    if reduction == "mean":
        schema_id = 0
        attr = b""
    elif reduction == "sum":
        schema_id = 0x4C45534D  # 'MSEL'
        attr = struct.pack("<i", 1)
    else:
        raise ValueError("only mean/sum supported in this test")

    def _run():
        _C.op_call(
            int(_C.OpKind.MseLoss),
            [pred, target],
            [out],
            schema_id,
            attr,
            0,
        )

    _run()
    diff = (out.item() - ref.item())
    adiff = abs(diff)
    tol = 1e-3 if dtype == torch.float16 else 1e-6
    status = "PASS" if adiff < tol else "FAIL"

    msg = f"[{name:<10}] Shape={str(tuple(shape)):<20} | Ref={ref.item():.6e} Out={out.item():.6e} | Diff={adiff:.2e} | {status}"

    if do_bench:
        ms = measure_loss_perf(_run)
        msg += f" | Time={ms:.4f} ms"

    print(msg)

def main():
    print(f"MseLoss enum value = {int(_C.OpKind.MseLoss)}")
    run_mse_loss_test((32, 1024), torch.float32, "F32-Mean", "mean")
    run_mse_loss_test((32, 1024), torch.float16, "F16-Mean", "mean")
    run_mse_loss_test((2048, 8192), torch.float32, "F32-Sum",  "sum")
    run_mse_loss_test((2048, 8192), torch.float16, "F16-Sum",  "sum")

if __name__ == "__main__":
    main()
