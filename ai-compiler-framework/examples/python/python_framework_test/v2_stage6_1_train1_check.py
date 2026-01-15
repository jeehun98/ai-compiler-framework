import copy
import torch
import sys
from pathlib import Path


THIS = Path(__file__).resolve()
EXAMPLES_PY = THIS.parents[1]
if str(EXAMPLES_PY) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_PY))

from aicf_fw.backend import set_backend, get_backend
from aicf_fw.core_v2 import (
    trace_ir, dump_ir,
    lower_to_backend_ops,
    build_binding_plan,
    dump_lowered, dump_plan,
)
from aicf_fw.core_v2.exec import PlannedExecutor, ExecOptions

# 너 프로젝트에서 쓰는 backend 생성 코드로 바꿔
from aicf_fw.backend.aicf_backend import AICFBackend  # 예시


def torch_reference_step(model, x, t, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
    # torch Adam 1-step
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(beta1, beta2), eps=eps)

    pred = model(x)
    loss = ((pred - t) ** 2).mean()
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=False)  # grads tensor 유지 (원하면)

    # 상태 꺼내기
    st = opt.state
    out = {}
    for name, p in model.named_parameters():
        out[f"param.{name}"] = p.detach().clone()
        # adam state
        s = st[p]
        out[f"m.{name}"] = s["exp_avg"].detach().clone()
        out[f"v.{name}"] = s["exp_avg_sq"].detach().clone()
    out["step"] = next(iter(st.values()))["step"].detach().clone()  # 보통 step 텐서/스칼라
    return out


def main():
    torch.manual_seed(0)
    device = torch.device("cuda")

    B, D = 64, 8
    x = torch.randn(B, D, device=device)
    t = torch.randn(B, D, device=device)

    # -------------------------
    # build torch model (same weights)
    # -------------------------
    torch_model = torch.nn.Sequential(
        torch.nn.Linear(D, D),
        torch.nn.ReLU(),
        torch.nn.Linear(D, D),
    ).to(device)
    torch_model.train()

    # clone for v2 path initial weights
    w0 = torch_model[0].weight.detach().clone()
    b0 = torch_model[0].bias.detach().clone()
    w1 = torch_model[2].weight.detach().clone()
    b1 = torch_model[2].bias.detach().clone()

    # -------------------------
    # reference (torch)
    # -------------------------
    ref = torch_reference_step(copy.deepcopy(torch_model), x, t, lr=1e-3)

    # -------------------------
    # core_v2 IR trace (너 stage6에서 쓰던 step_fn)
    # -------------------------
    def step_fn_v2():
        # 이 부분은 네 core_v2 ops로 작성된 train step 그대로 사용
        # 예:
        # y0 = linear(x, W0, b0)
        # r0 = relu(y0)
        # save(r0)
        # y1 = linear(r0, W1, b1)
        # dY = mse_grad(y1, t)
        # (bwd nodes)
        # step_inc, bias_corr, adam_step...
        raise NotImplementedError

    ir = trace_ir(step_fn_v2, name="v2_stage6_1_train1")
    lowered = lower_to_backend_ops(ir)
    plan = build_binding_plan(ir)

    print(dump_ir(ir))
    print(dump_lowered(lowered, title="LoweredOps(v2_stage6_1_train1)"))
    print(dump_plan(plan, title="BindingPlan(v2_stage6_1_train1)"))

    # -------------------------
    # backend + executor
    # -------------------------
    set_backend(AICFBackend())  # 너 프로젝트 방식대로

    ex = PlannedExecutor(
        ir=ir,
        lowered=lowered,
        plan=plan,
        opts=ExecOptions(debug=False),
        device=device,
    )

    # inputs/params 주입 (모델 파라미터만)
    params = {
        "0.W": w0.contiguous(),
        "0.b": b0.contiguous(),
        "2.W": w1.contiguous(),
        "2.b": b1.contiguous(),
    }
    inputs = {"x": x, "t": t}

    env = ex.run(inputs=inputs, params=params, reuse_static=True)

    # -------------------------
    # compare: params
    # -------------------------
    # v001: 0.W, v002:0.b, v006:2.W, v007:2.b 는 plan에서 param vid로 찾자
    name_to_vid = {s.name: s.vid for s in plan.specs.values()}

    def maxdiff(a, b):
        return float((a - b).abs().max().item())

    # torch uses weight shape (OUT, IN). 너도 그걸 사용 중이니 그대로 비교
    print("[cmp] W0", maxdiff(env[name_to_vid["0.W"]], ref["param.0.weight"]))
    print("[cmp] b0", maxdiff(env[name_to_vid["0.b"]], ref["param.0.bias"]))
    print("[cmp] W1", maxdiff(env[name_to_vid["2.W"]], ref["param.2.weight"]))
    print("[cmp] b1", maxdiff(env[name_to_vid["2.b"]], ref["param.2.bias"]))

    # optimizer state 비교도 가능(vid 이름이 opt.m.0.W 이런 식)
    print("[cmp] m0W", maxdiff(env[name_to_vid["opt.m.0.W"]], ref["m.0.weight"]))
    print("[cmp] v0W", maxdiff(env[name_to_vid["opt.v.0.W"]], ref["v.0.weight"]))
    print("[cmp] m1W", maxdiff(env[name_to_vid["opt.m.2.W"]], ref["m.2.weight"]))
    print("[cmp] v1W", maxdiff(env[name_to_vid["opt.v.2.W"]], ref["v.2.weight"]))

    print("DONE")


if __name__ == "__main__":
    main()
