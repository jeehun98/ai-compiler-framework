import os, sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_PY_DIR = REPO_ROOT / "examples" / "python"
PYMOD_DIR = REPO_ROOT / "build" / "python"
PKG_DIR   = PYMOD_DIR / "aicf_cuda"

sys.path.insert(0, str(EXAMPLES_PY_DIR))
sys.path.insert(0, str(PYMOD_DIR))
if os.name == "nt":
    os.add_dll_directory(str(PYMOD_DIR))
    os.add_dll_directory(str(PKG_DIR))

import torch
import aicf_cuda as aicf

torch.manual_seed(0)
device = "cuda"

def op_call(kind, inputs, outputs, attrs=None):
    aicf.op_call(kind, inputs, outputs, attrs or {})

def check_stats(tag, x):
    isn = torch.isnan(x).any().item()
    inf = torch.isinf(x).any().item()
    mx  = float(x.abs().max().item()) if x.numel() else 0.0
    print(f"[{tag}] nan={isn} inf={inf} absmax={mx}")

# ---------- data ----------
W_star = torch.randn(1024, 10, device=device) * 0.1
x = torch.randn(256, 1024, device=device).contiguous()
t = (x @ W_star).contiguous()

# ---------- init weights ----------
W0 = torch.empty(1024, 10, device=device, dtype=torch.float32)
torch.nn.init.kaiming_uniform_(W0, a=5**0.5)
W0 = W0.contiguous()
b0 = torch.zeros(10, device=device, dtype=torch.float32).contiguous()

# ---------- torch reference ----------
W_ref = W0.clone().detach().requires_grad_(True)
b_ref = b0.clone().detach().requires_grad_(True)
y_ref = x @ W_ref + b_ref
loss_ref = torch.mean((y_ref - t)**2)
loss_ref.backward()
dW_ref = W_ref.grad.detach().contiguous()
db_ref = b_ref.grad.detach().contiguous()

print("[torch] loss", float(loss_ref.item()),
      "dW_absmax", float(dW_ref.abs().max().item()),
      "db_absmax", float(db_ref.abs().max().item()))

# ---------- AICF: dY via MseGrad ----------
y = (x @ W0 + b0).contiguous()
dy = torch.empty_like(y).contiguous()

# scale=2/numel for mean MSE
scale = 2.0 / float(y.numel())
op_call(aicf.OpKind.MseGrad, [y, t], [dy], {"scale": float(scale)})
torch.cuda.synchronize()
check_stats("dy(aicf)", dy)

# ---------- AICF: dW = X^T @ dY ----------
# We need (1024,256) @ (256,10) => (1024,10)
xT = x.t().contiguous()
dW = torch.empty((xT.shape[0], dy.shape[1]), device=device, dtype=torch.float32).contiguous()

# Gemm contract in your backend may be "A @ B" on contiguous 2D.
# attrs/bias/act 없음.
op_call(aicf.OpKind.Gemm, [xT, dy], [dW], {})
torch.cuda.synchronize()
check_stats("dW(aicf)", dW)

# ---------- AICF: db = sum(dY over batch) ----------
# Want db shape (10,)
# Your ReduceSum variant reduces last-dim only and expects output rank1 [N].
# So transpose dy => (10,256), reduce last-dim (-1) -> (10,)
db = torch.empty((dy.shape[1],), device=device, dtype=torch.float32).contiguous()
op_call(aicf.OpKind.ReduceSum, [dy], [db], {"axis": int(-1)})  # or axis=1(last)

torch.cuda.synchronize()
check_stats("db(aicf)", db)

# ---------- compare ----------
dw_diff = float((dW - dW_ref).abs().max().item())
db_diff = float((db - db_ref).abs().max().item())
print("dW max abs diff =", dw_diff)
print("db max abs diff =", db_diff)

# quick sanity
print("dW_ref absmax", float(dW_ref.abs().max().item()))
print("db_ref absmax", float(db_ref.abs().max().item()))
