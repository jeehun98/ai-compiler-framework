from __future__ import annotations
import os, sys
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

from aicf_cuda import _C

def maxabs(x): return float(x.abs().max().item())
def maxabs_delta(a,b): return float((a-b).abs().max().item())

def main():
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    device = torch.device("cuda:0")
    torch.manual_seed(0)

    # param + grad + state
    p0 = torch.randn(8, 8, device=device, dtype=torch.float32).contiguous()
    g  = torch.randn(8, 8, device=device, dtype=torch.float32).contiguous()
    m0 = torch.zeros_like(p0)
    v0 = torch.zeros_like(p0)

    # host-managed meta
    bc1_inv = torch.tensor(1.0, device=device, dtype=torch.float32)
    bc2_inv = torch.tensor(1.0, device=device, dtype=torch.float32)

    # snapshots
    p_ref = p0.clone()
    m_ref = m0.clone()
    v_ref = v0.clone()

    attrs = {"lr": 1e-3, "beta1": 0.9, "beta2": 0.999, "eps": 1e-8}

    # call
    _C.op_call(int(_C.OpKind.AdamStep), [p0, g, m0, v0, bc1_inv, bc2_inv], [p0, m0, v0], attrs)

    print("[delta] |p|", maxabs_delta(p0, p_ref))
    print("[delta] |m|", maxabs_delta(m0, m_ref))
    print("[delta] |v|", maxabs_delta(v0, v_ref))
    print("[meta] bc1_inv", float(bc1_inv.item()), "bc2_inv", float(bc2_inv.item()))
    print("[grad] |g|", maxabs(g))

if __name__ == "__main__":
    main()
