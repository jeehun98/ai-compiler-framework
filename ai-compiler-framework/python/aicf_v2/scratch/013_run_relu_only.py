import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
import aicf_v2 as aicf

torch.manual_seed(0)

m = aicf.Model(dtype="f16", device="cuda")

x = m.input("x", aicf.TensorSpec(shape=(64, 256), dtype="f16", device="cuda"))

# relu 레이어로 IR emit
y = m.add(aicf.ReLU(name="r", save_for_bwd=False), x)
m.output("y", y)

exe = aicf.CudaExecutor()

feed = {
    "x": torch.randn((64, 256), device="cuda", dtype=torch.float16).contiguous()
}
ref = torch.relu(feed["x"])

out = exe.run(m, feed)
print("out keys =", list(out.keys()))

Y = out["y"]

print("max|delta| =", (Y - ref).abs().max().item())
