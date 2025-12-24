import os, sys
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[3]
PYMOD_DIR  = REPO_ROOT / "build" / "python"
PKG_DIR    = PYMOD_DIR / "aicf_cuda"
sys.path.insert(0, str(PYMOD_DIR))
if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import torch, aicf_cuda as aicf
torch.manual_seed(0)

def op_call(kind, inputs, outputs, attrs=None):
    aicf.op_call(kind, inputs, outputs, attrs or {})

lr = 1e-2
W0 = torch.randn(1<<20, device="cuda", dtype=torch.float32).contiguous()
dW = torch.randn_like(W0).contiguous()

W_a = W0.clone()
op_call(aicf.OpKind.SgdStep, [W_a, dW], [W_a], {"lr": float(lr)})
torch.cuda.synchronize()

W_t = (W0 - lr * dW).contiguous()

print("max abs diff =", float((W_a - W_t).abs().max().item()))
print("nan?", torch.isnan(W_a).any().item(), "absmax", float(W_a.abs().max().item()))
