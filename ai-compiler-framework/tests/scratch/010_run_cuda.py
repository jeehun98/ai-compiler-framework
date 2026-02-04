import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
import aicf_v2 as aicf

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

# 2) feed 준비 (inputs + params 이름 맞춰 공급)
# 주의: params는 IR에서 "fc1.W", "fc1.b" 이름으로 input_vids에 들어감
feed = {
    "x": torch.randn((1, 128), device="cuda", dtype=torch.float16),
    "fc1.W": torch.randn((256, 128), device="cuda", dtype=torch.float16),
    "fc1.b": torch.randn((256,), device="cuda", dtype=torch.float16),
}

# 3) 실행
exe = aicf.CudaExecutor()
out = exe.run(m, feed)

print("\n[OUT KEYS]", list(out.keys()))
print("[z]", out["res.out"].shape, out["res.out"].dtype, out["res.out"].device)
