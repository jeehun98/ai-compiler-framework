import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
import aicf_v2 as aicf

torch.manual_seed(0)

m = aicf.Model(dtype="f16", device="cuda")

a = m.input("a", aicf.TensorSpec(shape=(64, 256), dtype="f16", device="cuda"))
b = m.input("b", aicf.TensorSpec(shape=(64, 256), dtype="f16", device="cuda"))

y = m.add(aicf.Add(name="s"), a, b)
m.output("y", y)

exe = aicf.CudaExecutor()

feed = {
    "a": torch.randn((64, 256), device="cuda", dtype=torch.float16).contiguous(),
    "b": torch.randn((64, 256), device="cuda", dtype=torch.float16).contiguous(),
}
ref = feed["a"] + feed["b"]

out = exe.run(m, feed)
Y = out["y"]

print("max|delta| =", (Y - ref).abs().max().item())
