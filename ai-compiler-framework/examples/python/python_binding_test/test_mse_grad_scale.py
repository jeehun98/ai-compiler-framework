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

def op_call(kind, inputs, outputs, attrs=None):
    aicf.op_call(kind, inputs, outputs, attrs or {})

torch.manual_seed(0)
y = torch.randn(256,10, device="cuda", dtype=torch.float32).contiguous()
t = torch.randn_like(y).contiguous()
dy = torch.empty_like(y)

# 1) scale=0이면 dy는 0이어야 함
op_call(aicf.OpKind.MseGrad, [y,t], [dy], {"scale": float(0.0)})
torch.cuda.synchronize()
print("absmax(scale=0) =", float(dy.abs().max().item()))

# 2) scale=1이면 dy는 (y-t) 또는 2*(y-t) 중 하나 (커널 정의 확인용)
op_call(aicf.OpKind.MseGrad, [y,t], [dy], {"scale": float(1.0)})
torch.cuda.synchronize()
print("absmax(scale=1) =", float(dy.abs().max().item()))
