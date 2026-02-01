import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
import aicf_v2 as aicf

torch.manual_seed(0)
assert torch.cuda.is_available()

# ----------------------------
# 1) build model (IR)
# ----------------------------
m = aicf.Model(dtype="f16", device="cuda")

x = m.input("x", aicf.TensorSpec(shape=(1, 128), dtype="f16", device="cuda"))
y = m.add(aicf.Linear(128, 256, name="fc1", bias=True), x)
y = m.add(aicf.ReLU(name="relu1", save_for_bwd=False), y)
z = m.add(aicf.Add(name="res"), y, y)
m.output("z", z)

print(m.dump())

# ----------------------------
# 2) prepare feed
# ----------------------------
def make_feed(seed: int):
    g = torch.Generator(device="cuda")
    g.manual_seed(seed)

    feed = {
        "x": torch.randn((1, 128), device="cuda", dtype=torch.float16, generator=g),
        "fc1.W": torch.randn((256, 128), device="cuda", dtype=torch.float16, generator=g),
        "fc1.b": torch.randn((256,), device="cuda", dtype=torch.float16, generator=g),
    }
    return feed

feed0 = make_feed(123)

# ----------------------------
# 3) run eager vs cuda graph
# ----------------------------
exe = aicf.CudaExecutor()

# (A) eager run (no graph)
out_eager = exe.run(m, feed0, use_cuda_graph=False)
torch.cuda.synchronize()

# (B) first graph run (capture + replay)
out_g1 = exe.run(m, feed0, use_cuda_graph=True, mode="inference", warmup=2)
torch.cuda.synchronize()

# (C) second graph run (should be replay from cache)
out_g2 = exe.run(m, feed0, use_cuda_graph=True, mode="inference", warmup=2)
torch.cuda.synchronize()

print("\n[OUT KEYS eager]", list(out_eager.keys()))
print("[OUT KEYS graph ]", list(out_g1.keys()))

# ----------------------------
# 4) validate outputs
# ----------------------------
z_e = out_eager["z"]
z_1 = out_g1["z"]
z_2 = out_g2["z"]

print("\n[z] eager:", z_e.shape, z_e.dtype, z_e.device, "contig=", z_e.is_contiguous())
print("[z] graph1:", z_1.shape, z_1.dtype, z_1.device, "contig=", z_1.is_contiguous())
print("[z] graph2:", z_2.shape, z_2.dtype, z_2.device, "contig=", z_2.is_contiguous())

# exact equality는 fp16에서 연산 순서/커널 차이 있으면 깨질 수 있으니 allclose
ok_1 = torch.allclose(z_e, z_1, atol=1e-2, rtol=1e-2)
ok_2 = torch.allclose(z_1, z_2, atol=1e-2, rtol=1e-2)

print("\n[CHECK] eager vs graph1 allclose:", ok_1)
print("[CHECK] graph1 vs graph2 allclose:", ok_2)

if not ok_1:
    diff = (z_e.float() - z_1.float()).abs()
    print("[DIFF] max:", float(diff.max()), "mean:", float(diff.mean()))

# ----------------------------
# 5) change ONLY input x -> output should change
# (param fixed; typical inference)
# ----------------------------
feed1 = dict(feed0)
feed1["x"] = torch.randn((1, 128), device="cuda", dtype=torch.float16)  # new x only

out_g3 = exe.run(m, feed1, use_cuda_graph=True, mode="inference")
torch.cuda.synchronize()

changed = not torch.allclose(out_g2["z"], out_g3["z"], atol=1e-2, rtol=1e-2)
print("\n[CHECK] change x only -> z changed:", changed)

# ----------------------------
# 6) OPTIONAL: change weight too (should trigger new capture if you keep copy_roles=("input",) and static_roles includes param)
# - 현재 설정은 inference에서 copy_roles=("input",)라서 param 바꿔도 반영 안됨(그래프 내부 weight 고정)
# - weight 바꿔가며 재사용하고 싶으면 Executor에서 inference copy_roles에 "param" 추가해야 함.
# ----------------------------
feed2 = make_feed(999)
out_g4 = exe.run(m, feed2, use_cuda_graph=True, mode="inference")
torch.cuda.synchronize()

# 이 값은 "weight가 반영됐다"의 체크가 아니라,
# 현재 정책상 weight는 input처럼 copy하지 않으니 바뀌지 않을 수도 있음.
print("\n[NOTE] policy currently copies only role='input' on replay.")
print("       if you want W/b to update each run, set copy_roles to ('input','param') for inference.")
print("[DONE]")
