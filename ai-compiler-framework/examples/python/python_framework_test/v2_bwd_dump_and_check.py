from __future__ import annotations

import sys
from pathlib import Path
import torch

THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.backend.aicf_backend import AICFBackend
from aicf_fw.backend import set_backend

from aicf_fw.core_v2 import (
    trace_ir, dump_ir,
    lower_to_backend_ops, dump_lowered,
    build_binding_plan, dump_plan,
    PlannedExecutor, ExecOptions,
)
from aicf_fw.core_v2 import sym_tensor, linear, relu, mse_grad
from aicf_fw.core_v2.ops import save, linear_bwd, relu_bwd  # (2)에서 만든 것


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    dev = torch.device("cuda")
    dt = torch.float32

    bk = AICFBackend()
    set_backend(bk)

    # ---- trace graph (forward + mse_grad + backward) ----
    x = sym_tensor(name="x", shape=(64, 8), dtype=dt, device=dev)
    t = sym_tensor(name="t", shape=(64, 8), dtype=dt, device=dev)
    W0 = sym_tensor(name="0.W", shape=(8, 8), dtype=dt, device=dev)
    b0 = sym_tensor(name="0.b", shape=(8,), dtype=dt, device=dev)
    W1 = sym_tensor(name="2.W", shape=(8, 8), dtype=dt, device=dev)
    b1 = sym_tensor(name="2.b", shape=(8,), dtype=dt, device=dev)

    def step_fn():
        y0 = linear(x, W0, b0, name="lin0_out")
        a0 = relu(y0, name="relu0_out")
        saved_a0 = save(a0, name="relu0_saved")      # for relu_bwd
        y1 = linear(a0, W1, b1, name="lin1_out")
        dY = mse_grad(y1, t, name="dY")

        # ---- backward ----
        dA0, dW1, db1 = linear_bwd(a0, W1, dY, dx_name="d_relu0_out", dW_name="d_2.W", db_name="d_2.b")
        dY0 = relu_bwd(dA0, saved_a0, name="d_lin0_out")
        dX, dW0, db0 = linear_bwd(x, W0, dY0, dx_name="d_x", dW_name="d_0.W", db_name="d_0.b")

    ir = trace_ir(step_fn, name="v2_stage5_bwd")
    lowered = lower_to_backend_ops(ir)
    plan = build_binding_plan(ir)

    print(dump_ir(ir))
    print(dump_lowered(lowered, title="LoweredOps(v2_stage5_bwd)"))
    print(dump_plan(plan, title="BindingPlan(v2_stage5_bwd)"))

    # ---- runtime ----
    x_rt = torch.randn(64, 8, device=dev, dtype=dt)
    t_rt = torch.randn(64, 8, device=dev, dtype=dt)

    W0_rt = torch.randn(8, 8, device=dev, dtype=dt, requires_grad=True)
    b0_rt = torch.randn(8, device=dev, dtype=dt, requires_grad=True)
    W1_rt = torch.randn(8, 8, device=dev, dtype=dt, requires_grad=True)
    b1_rt = torch.randn(8, device=dev, dtype=dt, requires_grad=True)

    # planned exec uses plain tensors (no autograd)
    ex = PlannedExecutor(
        ir=ir, lowered=lowered, plan=plan, backend=bk, device=dev,
        opts=ExecOptions(debug=False),
    )

    env = ex.run(
        inputs={"x": x_rt, "t": t_rt},
        params={"0.W": W0_rt.detach(), "0.b": b0_rt.detach(), "2.W": W1_rt.detach(), "2.b": b1_rt.detach()},
        reuse_static=False,
    )
    torch.cuda.synchronize()

    # ---- torch ref ----
    y0_ref = x_rt @ W0_rt.t() + b0_rt
    a0_ref = torch.relu(y0_ref)
    y1_ref = a0_ref @ W1_rt.t() + b1_rt
    loss = ((y1_ref - t_rt) ** 2).mean()
    loss.backward()

    # find vids by name
    name_to_vid = {spec.name: vid for vid, spec in plan.specs.items()}
    dW0_v = env[int(name_to_vid["d_0.W"])]
    db0_v = env[int(name_to_vid["d_0.b"])]
    dW1_v = env[int(name_to_vid["d_2.W"])]
    db1_v = env[int(name_to_vid["d_2.b"])]

    def md(a, b): return (a - b).abs().max().item()

    print("[check] dW0 maxdiff =", md(dW0_v, W0_rt.grad))
    print("[check] db0 maxdiff =", md(db0_v, b0_rt.grad))
    print("[check] dW1 maxdiff =", md(dW1_v, W1_rt.grad))
    print("[check] db1 maxdiff =", md(db1_v, b1_rt.grad))

    print("OK")


if __name__ == "__main__":
    main()
