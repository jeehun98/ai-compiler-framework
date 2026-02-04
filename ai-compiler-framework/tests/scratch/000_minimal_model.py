import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import aicf_v2 as aicf

m = aicf.Model(dtype="f16", device="cuda")

x = m.input("x", aicf.TensorSpec(shape=(1, 128), dtype="f16", device="cuda"))

y = m.add(aicf.Linear(128, 256, name="fc1"), x)
y = m.add(aicf.ReLU(name="relu1", save_for_bwd=False), y)
z = m.add(aicf.Add(name="res"), y, y)

m.output("z", z)

print(m.dump())
