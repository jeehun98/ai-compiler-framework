import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import aicf_v2 as aicf

def print_values(m: aicf.Model):
    b = m.b
    print("\n[VALUES]")
    for v in b.values:
        s = v.spec
        print(f"  v{v.vid:02d}  {v.name:20s}  {s.dtype} {s.device} shape={s.shape}  prod={v.producer_op} users={v.users}")

def print_ops(m: aicf.Model):
    b = m.b
    print("\n[OPS]")
    for i, op in enumerate(b.ops):
        ins = ", ".join(f"v{vid}:{b.values[vid].name}" for vid in op.inputs)
        outs = ", ".join(f"v{vid}:{b.values[vid].name}" for vid in op.outputs)
        print(f"  o{i:02d} {op.kind:10s} {op.name:20s}  ({ins}) -> ({outs})")
        if op.attrs:
            print(f"       attrs={op.attrs}")
        if op.constraints:
            print(f"       constraints={op.constraints}")
        if op.hints:
            print(f"       hints={op.hints}")
        if op.saved:
            print(f"       saved={[b.values[v].name for v in op.saved]}")

m = aicf.Model(dtype="f16", device="cuda")

x = m.input("x", aicf.TensorSpec(shape=(1, 128), dtype="f16", device="cuda"))
y = m.add(aicf.Linear(128, 256, name="fc1"), x)
y = m.add(aicf.ReLU(name="relu1", save_for_bwd=True), y)
z = m.add(aicf.Add(name="res"), y, y)
m.output("z", z)

print(m.dump())
print_values(m)
print_ops(m)
