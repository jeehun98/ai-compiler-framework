import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import aicf_v2 as aicf

def step(title: str, m: aicf.Model):
    print("\n" + "=" * 80)
    print(title)
    print("-" * 80)
    print(m.dump())

m = aicf.Model(dtype="f16", device="cuda")

# 1) input
x = m.input("x", aicf.TensorSpec(shape=(1, 128), dtype="f16", device="cuda"))
step("[1] after input(x)", m)

# 2) linear
y = m.add(aicf.Linear(128, 256, name="fc1"), x)
step("[2] after fc1(x)", m)

# 3) relu
y = m.add(aicf.ReLU(name="relu1", save_for_bwd=False), y)
step("[3] after relu1(y)", m)

# 4) add/residual
z = m.add(aicf.Add(name="res"), y, y)
step("[4] after res(y, y)", m)

# 5) output
m.output("z", z)
step("[5] after output(z)", m)
