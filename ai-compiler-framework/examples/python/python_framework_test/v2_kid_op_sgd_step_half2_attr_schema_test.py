from __future__ import annotations

import sys
from pathlib import Path
import struct
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

import _C  # build/python/aicf_cuda/_C.pyd 로딩 경로가 잡혀있어야 함

KIND = _C.OpKind.SgdStep
SCHEMA_SGDS = 0x53444753  # 'SGDS'

def infer_lr(p0: torch.Tensor, p1: torch.Tensor, g: torch.Tensor) -> float:
    # lr ~= mean(|p0-p1|) / mean(|g|)
    num = (p0 - p1).abs().mean().item()
    den = g.abs().mean().item() + 1e-12
    return float(num / den)

def main():
    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.float16

    # half2 조건: numel even + 4B align (보통 contiguous f16이면 4B align 잘 나옴)
    n = 4096  # even
    p = torch.randn(n, device=device, dtype=dtype)
    g = torch.randn(n, device=device, dtype=dtype)

    # reference
    lr = 1e-2
    p_ref = p - lr * g

    # run kernel (in-place)
    p_out = p.clone()

    attrs = struct.pack("<f", lr)

    _C.launch_by_id(
        "sgd_step_f16_half2_v0",
        KIND,
        [p_out, g],
        [p_out],
        SCHEMA_SGDS,
        attrs,
        0,
    )

    lr_est = infer_lr(p, p_out, g)
    print(f"[probe] inferred_lr~{lr_est:.8f} (target={lr})")
    # 넉넉히 허용 (FP16/half2 라운딩)
    assert abs(lr_est - lr) < 2e-3, "lr attr not applied (schema or attrs_bytes broken?)"

    # allclose도 같이
    max_abs = (p_out - p_ref).abs().max().item()
    print(f"[probe] max_abs_diff={max_abs}")
    assert max_abs < 5e-2, "sgd_step half2 output mismatch too large"

    print("OK")

if __name__ == "__main__":
    main()
