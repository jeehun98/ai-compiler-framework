from __future__ import annotations
import sys, os
from typing import Dict, List, Tuple

import torch

# ----------------------------
# path setup (your usual)
# ----------------------------
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import aicf_v2 as aicf  # noqa


torch.manual_seed(0)
assert torch.cuda.is_available()


# ----------------------------
# 1) Build graph:
#   bc1,bc2 = bias_corr(step)
#   (P,M,V) = adam_step(w, g, m, v, bc1, bc2)
#   step = step_inc(step)
# ----------------------------
m = aicf.Model(dtype="f32", device="cuda")

# externals
w = m.param("w", aicf.TensorSpec(shape=(256, 128), dtype="f32", device="cuda"))
g = m.input("g", aicf.TensorSpec(shape=(256, 128), dtype="f32", device="cuda"))
m1 = m.state("m", aicf.TensorSpec(shape=(256, 128), dtype="f32", device="cuda"))
v1 = m.state("v", aicf.TensorSpec(shape=(256, 128), dtype="f32", device="cuda"))

# step: v2 policy may want (1,) instead of 0d
# your BiasCorr expects step shape (1,)
step = m.state("step", aicf.TensorSpec(shape=(1,), dtype="i32", device="cuda"))

bc1, bc2 = m.add(aicf.BiasCorr(name="biascorr", beta1=0.9, beta2=0.999), step)
P2, M2, V2 = m.add(
    aicf.AdamStep(name="adam", lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8),
    w, g, m1, v1, bc1, bc2
)
step2 = m.add(aicf.StepInc(name="step_inc"), step)

m.output("adam.P", P2)
m.output("adam.M", M2)
m.output("adam.V", V2)
m.output("step_inc.out", step2)

print(m.dump())

# debug vids
b = m.b
print("[DEBUG] input_vids   :", [b.values[v].name for v in b.input_vids])
print("[DEBUG] external_vids:", [b.values[v].name for v in b.external_vids])
print("[DEBUG] param_vids   :", [b.values[v].name for v in getattr(b, "param_vids", [])])
print("[DEBUG] state_vids   :", [b.values[v].name for v in getattr(b, "state_vids", [])])


# ----------------------------
# 2) helpers
# ----------------------------
def make_feed(seed: int) -> Dict[str, torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    return {
        "w": torch.randn((256, 128), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
        "g": torch.randn((256, 128), device="cuda", dtype=torch.float32, generator=gen).contiguous(),
        "m": torch.zeros((256, 128), device="cuda", dtype=torch.float32).contiguous(),
        "v": torch.zeros((256, 128), device="cuda", dtype=torch.float32).contiguous(),
        "step": torch.zeros((1,), device="cuda", dtype=torch.int32).contiguous(),
    }


def make_grads(steps: int, seed: int) -> List[torch.Tensor]:
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    return [
        torch.randn((256, 128), device="cuda", dtype=torch.float32, generator=gen).contiguous()
        for _ in range(steps)
    ]


@torch.no_grad()
def run_steps(
    exe: aicf.CudaExecutor,
    feed0: Dict[str, torch.Tensor],
    grads: List[torch.Tensor],
    *,
    use_cuda_graph: bool,
) -> Tuple[List[torch.Tensor], List[int]]:
    Ps: List[torch.Tensor] = []
    steps_out: List[int] = []

    # NOTE: keep base buffers separate per executor run
    feed_base = {k: v.clone() for k, v in feed0.items()}

    for gi in grads:
        feed_i = dict(feed_base)
        feed_i["g"] = gi  # fixed sequence

        out = exe.run(
            m,
            feed_i,
            use_cuda_graph=use_cuda_graph,
            mode="train",
            warmup=0,
        )
        torch.cuda.synchronize()

        Ps.append(out["adam.P"].clone())
        steps_out.append(int(out["step_inc.out"].view(-1)[0].item()))

    return Ps, steps_out


# ----------------------------
# 3) run eager vs graph with SAME grads
# ----------------------------
exe_eager = aicf.CudaExecutor()
exe_graph = aicf.CudaExecutor()

feed0_eager = make_feed(123)
feed0_graph = {k: v.clone() for k, v in feed0_eager.items()}

grads = make_grads(steps=3, seed=9999)

Ps_e, ss_e = run_steps(exe_eager, feed0_eager, grads, use_cuda_graph=False)
Ps_g, ss_g = run_steps(exe_graph, feed0_graph, grads, use_cuda_graph=True)

print("\n[EAGER] steps:", ss_e)
print("[GRAPH] steps:", ss_g)
print("\n[CHECK] step sequences equal:", ss_e == ss_g)

for i in range(3):
    ok = torch.allclose(Ps_e[i], Ps_g[i], atol=1e-2, rtol=1e-2)
    print(f"[CHECK] step {i+1} w allclose:", ok)
    if not ok:
        diff = (Ps_e[i] - Ps_g[i]).abs()
        print("        diff max:", float(diff.max()), "mean:", float(diff.mean()))


# ----------------------------
# 4) attack test: train replay must not overwrite param/state from feed
# ----------------------------
feed_attack = make_feed(2026)
grads_attack = make_grads(steps=2, seed=7)

# first run: establish internal state
out1 = exe_graph.run(
    m,
    {**feed_attack, "g": grads_attack[0]},
    use_cuda_graph=True,
    mode="train",
    warmup=0,
)
torch.cuda.synchronize()
P_before = out1["adam.P"].clone()
s_before = int(out1["step_inc.out"].view(-1)[0].item())

# attempt to reset w/m/v/step via feed (should be ignored on replay; only input 'g' copied)
feed_attack2 = dict(feed_attack)
feed_attack2["g"] = grads_attack[1]
feed_attack2["w"] = torch.zeros_like(feed_attack2["w"])
feed_attack2["m"] = torch.zeros_like(feed_attack2["m"])
feed_attack2["v"] = torch.zeros_like(feed_attack2["v"])
feed_attack2["step"] = torch.zeros_like(feed_attack2["step"])

out2 = exe_graph.run(
    m,
    feed_attack2,
    use_cuda_graph=True,
    mode="train",
    warmup=0,
)
torch.cuda.synchronize()
P_after = out2["adam.P"].clone()
s_after = int(out2["step_inc.out"].view(-1)[0].item())

step_ok = (s_after > s_before)
not_reset = not torch.allclose(P_after, torch.zeros_like(P_after), atol=1e-2, rtol=1e-2)

print("\n[ATTACK TEST]")
print("  step before/after:", s_before, "->", s_after, " step_ok:", step_ok)
print("  w not reset:", not_reset)

print("\n[DONE]")
