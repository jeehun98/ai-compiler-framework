import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
import aicf_v2 as aicf

torch.manual_seed(0)
assert torch.cuda.is_available()

# ----------------------------
# 1) Build a minimal "train-like" graph:
#    w(param) updated by sgd_step using g(input)
#    step(state) incremented by step_inc
#
# IMPORTANT:
# - StepInc / SgdStep should be in-place emit (outputs=[input_vid])
#   OR plan should alias out->in.
# - This test assumes you already changed layers to in-place emit.
# ----------------------------
m = aicf.Model(dtype="f16", device="cuda")

g = m.input("g", aicf.TensorSpec(shape=(256, 128), dtype="f16", device="cuda"))
w = m.param("w", aicf.TensorSpec(shape=(256, 128), dtype="f16", device="cuda"))
step = m.state("step", aicf.TensorSpec(shape=(), dtype="i32", device="cuda"))

w2 = m.add(aicf.SgdStep(name="sgd_w", lr=1e-3), w, g)
if isinstance(w2, tuple):
    w2 = w2[0]

step2 = m.add(aicf.StepInc(name="step_inc"), step)
if isinstance(step2, tuple):
    step2 = step2[0]

m.output("w", w2)
m.output("step", step2)

print(m.dump())

# Debug: confirm builder separation is correct
b = m.b
print("[DEBUG] input_vids  :", [b.values[v].name for v in b.input_vids])
print("[DEBUG] external_vids:", [b.values[v].name for v in b.external_vids])
print("[DEBUG] param_vids  :", [b.values[v].name for v in getattr(b, "param_vids", [])])
print("[DEBUG] state_vids  :", [b.values[v].name for v in getattr(b, "state_vids", [])])

# ----------------------------
# 2) helper: make feed
# ----------------------------
def make_feed(seed: int):
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    return {
        "g": torch.randn((256, 128), device="cuda", dtype=torch.float16, generator=gen),
        "w": torch.randn((256, 128), device="cuda", dtype=torch.float16, generator=gen),
        "step": torch.zeros((), device="cuda", dtype=torch.int32),
    }

# ----------------------------
# 3) IMPORTANT: compile ONCE per executor
#    (if you compile every step, plan identity changes -> graph cache miss -> step resets)
# ----------------------------
exe_eager = aicf.CudaExecutor()
exe_graph = aicf.CudaExecutor()

prog_eager = exe_eager.compile(m)
prog_graph = exe_graph.compile(m)

feed0_eager = make_feed(123)
feed0_graph = {
    "g": feed0_eager["g"].clone(),
    "w": feed0_eager["w"].clone(),
    "step": feed0_eager["step"].clone(),
}

def run_steps(exe, prog, feed, *, use_cuda_graph: bool, steps: int):
    ws = []
    steps_out = []
    for i in range(steps):
        # new gradient each iteration
        feed_i = dict(feed)
        feed_i["g"] = torch.randn_like(feed_i["g"])

        out = exe.run_compiled(
            m,
            prog,
            feed_i,
            use_cuda_graph=use_cuda_graph,
            mode="train",
            warmup=0,   # make step progression obvious
        )
        torch.cuda.synchronize()

        ws.append(out["w"].clone())
        steps_out.append(int(out["step"].item()))

    return ws, steps_out

ws_e, st_e = run_steps(exe_eager, prog_eager, feed0_eager, use_cuda_graph=False, steps=3)
ws_g, st_g = run_steps(exe_graph, prog_graph, feed0_graph, use_cuda_graph=True, steps=3)

print("\n[EAGER] steps:", st_e)
print("[GRAPH] steps:", st_g)

print("\n[CHECK] step sequences equal:", st_e == st_g)

for i in range(3):
    ok = torch.allclose(ws_e[i], ws_g[i], atol=1e-2, rtol=1e-2)
    print(f"[CHECK] w step {i+1} eager vs graph allclose:", ok)
    if not ok:
        diff = (ws_e[i].float() - ws_g[i].float()).abs()
        print("        diff max:", float(diff.max()), "mean:", float(diff.mean()))

# ----------------------------
# 4) ATTACK TEST:
#    In train mode, param/state should NOT be overwritten from feed on replay.
#    - We try to reset w/step via feed; should be ignored.
# ----------------------------
feed_attack = make_feed(999)

out1 = exe_graph.run_compiled(
    m, prog_graph, feed_attack,
    use_cuda_graph=True, mode="train", warmup=0
)
torch.cuda.synchronize()
w_before = out1["w"].clone()
s_before = int(out1["step"].item())

feed_attack2 = dict(feed_attack)
feed_attack2["g"] = torch.randn_like(feed_attack2["g"])
feed_attack2["w"] = torch.zeros_like(feed_attack2["w"])        # attempt reset param
feed_attack2["step"] = torch.zeros_like(feed_attack2["step"])  # attempt reset state

out2 = exe_graph.run_compiled(
    m, prog_graph, feed_attack2,
    use_cuda_graph=True, mode="train", warmup=0
)
torch.cuda.synchronize()

w_after = out2["w"].clone()
s_after = int(out2["step"].item())

step_ok = (s_after > s_before)
not_reset = not torch.allclose(w_after, torch.zeros_like(w_after), atol=1e-2, rtol=1e-2)

print("\n[ATTACK TEST]")
print("  step before/after:", s_before, "->", s_after, " step_ok:", step_ok)
print("  w not reset to zeros:", not_reset)

print("\n[DONE]")
