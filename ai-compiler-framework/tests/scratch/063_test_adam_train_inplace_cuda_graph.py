from __future__ import annotations
from typing import Dict, List, Tuple

import torch
import sys
from pathlib import Path

p = Path(__file__).resolve()
root = None
for parent in [p] + list(p.parents):
    if (parent / "pyproject.toml").exists():
        root = parent
        break
if root is None:
    raise RuntimeError("pyproject.toml not found")

py_src = root / "python" / "aicf_v2" / "src"
sys.path.insert(0, str(py_src))

import aicf_v2 as aicf
print("Imported:", aicf.__file__)



torch.manual_seed(0)
assert torch.cuda.is_available()


# ----------------------------
# 1) Build graph (inplace via plan.alias)
#   bc1,bc2 = bias_corr(step)
#   (Pout,Mout,Vout) = adam_step(w, g, m, v, bc1, bc2)   with alias => w,m,v updated
#   step = step_inc(step)
# ----------------------------
m = aicf.Model(dtype="f32", device="cuda")

w = m.param("w", aicf.TensorSpec(shape=(256, 128), dtype="f32", device="cuda"))
g = m.input("g", aicf.TensorSpec(shape=(256, 128), dtype="f32", device="cuda"))
m1 = m.state("m", aicf.TensorSpec(shape=(256, 128), dtype="f32", device="cuda"))
v1 = m.state("v", aicf.TensorSpec(shape=(256, 128), dtype="f32", device="cuda"))

# BiasCorr in v2: step shape (1,)
step = m.state("step", aicf.TensorSpec(shape=(1,), dtype="i32", device="cuda"))

bc1, bc2 = m.add(aicf.BiasCorr(name="biascorr", beta1=0.9, beta2=0.999), step)
P2, M2, V2 = m.add(
    aicf.AdamStep(name="adam", lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8),
    w, g, m1, v1, bc1, bc2
)
step2 = m.add(aicf.StepInc(name="step_inc"), step)

# IMPORTANT:
# after plan.alias:
#   slots[P2] points to slots[w]
#   slots[M2] points to slots[m]
#   slots[V2] points to slots[v]
# so exporting w/m/v is the "real" updated state view.
m.output("w", w)
m.output("m", m1)
m.output("v", v1)
m.output("step", step)

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
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[int]]:
    Ws: List[torch.Tensor] = []
    Ms: List[torch.Tensor] = []
    Vs: List[torch.Tensor] = []
    Ss: List[int] = []

    # keep base buffers separate
    feed_base = {k: v.clone() for k, v in feed0.items()}

    for gi in grads:
        feed_i = dict(feed_base)
        feed_i["g"] = gi  # fixed grad sequence

        out = exe.run(
            m,
            feed_i,
            use_cuda_graph=use_cuda_graph,
            mode="train",
            warmup=0,
        )
        torch.cuda.synchronize()

        Ws.append(out["w"].clone())
        Ms.append(out["m"].clone())
        Vs.append(out["v"].clone())
        Ss.append(int(out["step"].view(-1)[0].item()))

    return Ws, Ms, Vs, Ss


# ----------------------------
# 3) run eager vs graph with SAME grads
# ----------------------------
exe_eager = aicf.CudaExecutor()
exe_graph = aicf.CudaExecutor()

feed0_eager = make_feed(123)
feed0_graph = {k: v.clone() for k, v in feed0_eager.items()}

grads = make_grads(steps=3, seed=9999)

W_e, M_e, V_e, S_e = run_steps(exe_eager, feed0_eager, grads, use_cuda_graph=False)
W_g, M_g, V_g, S_g = run_steps(exe_graph, feed0_graph, grads, use_cuda_graph=True)

print("\n[EAGER] steps:", S_e)
print("[GRAPH] steps:", S_g)
print("\n[CHECK] step sequences equal:", S_e == S_g)

for i in range(3):
    okw = torch.allclose(W_e[i], W_g[i], atol=1e-2, rtol=1e-2)
    okm = torch.allclose(M_e[i], M_g[i], atol=1e-2, rtol=1e-2)
    okv = torch.allclose(V_e[i], V_g[i], atol=1e-2, rtol=1e-2)
    print(f"[CHECK] step {i+1} W allclose:", okw, " M:", okm, " V:", okv)
    if not (okw and okm and okv):
        if not okw:
            d = (W_e[i] - W_g[i]).abs()
            print("        W diff max:", float(d.max()), "mean:", float(d.mean()))
        if not okm:
            d = (M_e[i] - M_g[i]).abs()
            print("        M diff max:", float(d.max()), "mean:", float(d.mean()))
        if not okv:
            d = (V_e[i] - V_g[i]).abs()
            print("        V diff max:", float(d.max()), "mean:", float(d.mean()))

# also check that state actually changes across steps (sanity)
chg_w = not torch.allclose(W_g[0], W_g[-1], atol=1e-6, rtol=1e-6)
chg_m = not torch.allclose(M_g[0], M_g[-1], atol=1e-6, rtol=1e-6)
chg_v = not torch.allclose(V_g[0], V_g[-1], atol=1e-6, rtol=1e-6)
print("\n[SANITY] state changes across steps (graph):",
      "W:", chg_w, "M:", chg_m, "V:", chg_v)

# ----------------------------
# 4) ATTACK TEST:
#    train replay must not overwrite param/state from feed on replay.
# ----------------------------
feed_attack = make_feed(2026)
grads_attack = make_grads(steps=2, seed=7)

out1 = exe_graph.run(
    m,
    {**feed_attack, "g": grads_attack[0]},
    use_cuda_graph=True,
    mode="train",
    warmup=0,
)
torch.cuda.synchronize()
w_before = out1["w"].clone()
s_before = int(out1["step"].view(-1)[0].item())

feed_attack2 = dict(feed_attack)
feed_attack2["g"] = grads_attack[1]
feed_attack2["w"] = torch.zeros_like(feed_attack2["w"])        # attempt reset param
feed_attack2["m"] = torch.zeros_like(feed_attack2["m"])        # attempt reset state
feed_attack2["v"] = torch.zeros_like(feed_attack2["v"])        # attempt reset state
feed_attack2["step"] = torch.zeros_like(feed_attack2["step"])  # attempt reset state

out2 = exe_graph.run(
    m,
    feed_attack2,
    use_cuda_graph=True,
    mode="train",
    warmup=0,
)
torch.cuda.synchronize()

w_after = out2["w"].clone()
s_after = int(out2["step"].view(-1)[0].item())

step_ok = (s_after > s_before)
not_reset = not torch.allclose(w_after, torch.zeros_like(w_after), atol=1e-2, rtol=1e-2)

print("\n[ATTACK TEST]")
print("  step before/after:", s_before, "->", s_after, " step_ok:", step_ok)
print("  w not reset:", not_reset)

print("\n[DONE]")
