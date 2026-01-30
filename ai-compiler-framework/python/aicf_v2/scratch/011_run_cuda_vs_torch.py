import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
import aicf_v2 as aicf

def maxabs(a, b):
    return (a - b).abs().max().item()

torch.manual_seed(0)
assert torch.cuda.is_available()

# 1) IR 빌드
m = aicf.Model(dtype="f16", device="cuda")

x_vid = m.input("x", aicf.TensorSpec(shape=(1, 128), dtype="f16", device="cuda"))
y = m.add(aicf.Linear(128, 256, name="fc1", bias=True), x_vid)
y = m.add(aicf.ReLU(name="relu1", save_for_bwd=False), y)
z = m.add(aicf.Add(name="res"), y, y)
m.output("z", z)

print(m.dump())

# 2) feed
X = torch.randn((1, 128), device="cuda", dtype=torch.float16).contiguous()
W = torch.randn((256, 128), device="cuda", dtype=torch.float16).contiguous()
b = torch.randn((256,), device="cuda", dtype=torch.float16).contiguous()

feed = {"x": X, "fc1.W": W, "fc1.b": b}

# 3) v2 실행
exe = aicf.CudaExecutor()
out = exe.run(m, feed)
Z = out["res.out"]

# 4) torch ref
# ref_fp16: torch가 내부 정책으로 fp16/tc 처리 (커널과 더 유사)
ref_fp16 = torch.relu(X @ W.t() + b) + torch.relu(X @ W.t() + b)

# ref_fp32: fp32 accumulate 기준 (더 엄격하지만 차이 날 수 있음)
ref_fp32 = (torch.relu((X.float() @ W.float().t()) + b.float()).half() * 2)

d16 = maxabs(Z, ref_fp16)
d32 = maxabs(Z, ref_fp32)

print("\nmax|delta| vs torch-fp16-ref =", d16)
print("max|delta| vs torch-fp32-ref =", d32)
print("Z:", Z.shape, Z.dtype, Z.device)
