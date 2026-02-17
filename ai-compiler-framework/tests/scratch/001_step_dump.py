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
