import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import torch
import aicf_v2 as aicf

from aicf_v2.emitters.cuda.context import CudaEmitContext
from aicf_v2.emitters.cuda.bias_add import bias_add as emit_bias_add


def torch_bias_add_ref(Y, bias, axis=-1):
    r = Y.dim()
    if axis == -1:
        axis = r - 1
    shape = [1] * r
    shape[axis] = bias.numel()
    return Y + bias.view(*shape)


torch.manual_seed(0)

m = aicf.Model(dtype="f16", device="cuda")
Y = m.input("Y", aicf.TensorSpec(shape=(64, 256), dtype="f16", device="cuda"))
b = m.input("b", aicf.TensorSpec(shape=(256,), dtype="f16", device="cuda"))

out = m.b.value("O", aicf.TensorSpec(shape=(64, 256), dtype="f16", device="cuda"))

# ✅ emitter-first: use resolved emitter (fills kind_id/schema/blob)
ctx = CudaEmitContext()
emit_bias_add(
    m.b, ctx,
    x=Y,
    bias=b,
    out=out,
    broadcast_axis=-1,
    name="test.bias_add",
    constraints={"inplace_ok": True},
)

m.output("O", out)

exe = aicf.CudaExecutor()

feed = {
    "Y": torch.randn((64, 256), device="cuda", dtype=torch.float16).contiguous(),
    "b": torch.randn((256,), device="cuda", dtype=torch.float16).contiguous(),
}
ref = torch_bias_add_ref(feed["Y"].float(), feed["b"].float(), -1).half()

outd = exe.run(m, feed)
O = outd["O"]

print("max|delta| =", (O - ref).abs().max().item())
