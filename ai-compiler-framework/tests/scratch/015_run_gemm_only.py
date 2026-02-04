import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
import aicf_v2 as aicf

from aicf_v2.emitters.cuda.context import CudaEmitContext
from aicf_v2.emitters.cuda.gemm import gemm as emit_gemm

torch.manual_seed(0)

m = aicf.Model(dtype="f16", device="cuda")

x = m.input("x", aicf.TensorSpec(shape=(64, 128), dtype="f16", device="cuda"))
W = m.input("W", aicf.TensorSpec(shape=(256, 128), dtype="f16", device="cuda"))  # (N,K)

out = m.b.value("Y", aicf.TensorSpec(shape=(64, 256), dtype="f16", device="cuda"))

# ✅ emitter-first
ctx = CudaEmitContext()
emit_gemm(m.b, ctx, A=x, B=W, out=out, transA=False, transB=True, name="g.gemm")


m.output("Y", out)

exe = aicf.CudaExecutor()

X = torch.randn((64, 128), device="cuda", dtype=torch.float16).contiguous()
WT = torch.randn((256, 128), device="cuda", dtype=torch.float16).contiguous()

# ref: X @ WT.T
ref = (X.float() @ WT.float().t()).half()

outd = exe.run(m, {"x": X, "W": WT})
Y = outd["Y"]

print("max|delta| =", (Y - ref).abs().max().item())
