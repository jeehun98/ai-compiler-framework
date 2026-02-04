import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import aicf_v2 as aicf

m = aicf.Model(dtype="f16", device="cuda")

x = m.input("x", aicf.TensorSpec(shape=(1, 128), dtype="f16", device="cuda"))
y = m.add(aicf.Linear(128, 256, name="fc1"), x)

y = m.add(aicf.ReLU(name="relu1", save_for_bwd=True), y)

# relu op 찾기
b = m.b
relu_ops = [op for op in b.ops if op.kind == "relu"]
assert len(relu_ops) == 1, f"expected 1 relu op, got {len(relu_ops)}"
relu = relu_ops[0]

print(m.dump())
print("\n[CHECK] relu.saved =", [b.values[v].name for v in relu.saved])
assert len(relu.saved) == 1, "relu saved should have 1 value"
print("[OK] saved policy recorded in IR")
