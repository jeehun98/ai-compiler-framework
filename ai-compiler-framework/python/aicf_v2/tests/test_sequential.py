# aicf_v2/tests/test_sequential.py
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


import torch

from aicf_v2.fw.emit_ctx import EmitCtx
from aicf_v2.models.sequential import Sequential
from aicf_v2.nn.linear import Linear
from aicf_v2.backend.executor_torch import TorchExecutor

def test_sequential_linear_torch_exec():
    torch.manual_seed(0)

    # model (2-layer MLP)
    model = Sequential(layers=[Linear(32, 16, bias=True), Linear(16, 8, bias=False)])

    # compile (IR + lowered 동시 생성)
    ctx = EmitCtx(B=4, dtype="f16", device="cuda")

    x_vid = ctx.new_value("x", shape=(ctx.B, 32))
    y_vid = model.emit(ctx, x_vid)

    # ---- prepare real tensors for provided buffers ----
    # EmitCtx에서는 param name -> vid를 내부에 저장했으니
    # 여기선 prog.value_names를 이용해 param들을 찾아서 채워주자.
    # (v0 방식: 이름 기반으로 파라미터 버퍼 찾아 주입)
    name_to_vid = {name: vid for vid, name in enumerate(ctx.prog.value_names)}
    vid_to_bid = ctx.prog.value_to_buffer

    # inputs
    x = torch.randn((ctx.B, 32), device="cuda", dtype=torch.float16)
    provided = {vid_to_bid[x_vid]: x}

    # params: "0.W", "0.b", "1.W" 가 있어야 함
    W0 = torch.randn((16, 32), device="cuda", dtype=torch.float16)
    b0 = torch.randn((16,), device="cuda", dtype=torch.float16)
    W1 = torch.randn((8, 16), device="cuda", dtype=torch.float16)

    provided[vid_to_bid[name_to_vid["0.W"]]] = W0
    provided[vid_to_bid[name_to_vid["0.b"]]] = b0
    provided[vid_to_bid[name_to_vid["1.W"]]] = W1

    # ---- run lowered with TorchExecutor ----
    exe = TorchExecutor(debug=False)
    outs, _bufs = exe.run(ctx.prog, provided=provided)
    y_aicf = outs["out0"]

    # ---- torch reference ----
    y_ref = (x @ W0.t()) + b0
    y_ref = (y_ref @ W1.t())

    # ---- check ----
    torch.testing.assert_close(y_aicf, y_ref, rtol=1e-3, atol=1e-3)

    # sanity: shape matches compile-time spec
    assert ctx.prog.value_specs[y_vid].shape == tuple(y_aicf.shape)