import sys, os
sys.path.append(os.path.abspath(os.path.join(__file__, "..", "..", "src")))

import aicf_v2 as aicf

def sanity(m: aicf.Model):
    b = m.b

    # vids 유효성
    nvals = len(b.values)
    for op in b.ops:
        for v in op.inputs + op.outputs + op.saved:
            assert 0 <= v < nvals, f"invalid vid {v} (nvals={nvals})"

    # input_vids / output_vids 유효성
    for v in b.input_vids + b.output_vids:
        assert 0 <= v < nvals, f"invalid io vid {v}"

    # producer/users 간단 체크
    for op_idx, op in enumerate(b.ops):
        for out in op.outputs:
            assert b.values[out].producer_op == op_idx, "producer_op mismatch"

    # Linear shape 체크 예시: 마지막 output이 (1,256)인지
    out_vid = b.output_vids[-1]
    out_shape = b.values[out_vid].spec.shape
    assert out_shape == (1, 256), f"unexpected final out shape: {out_shape}"

    print("[OK] sanity checks passed")

m = aicf.Model(dtype="f16", device="cuda")

x = m.input("x", aicf.TensorSpec(shape=(1, 128), dtype="f16", device="cuda"))
y = m.add(aicf.Linear(128, 256, name="fc1"), x)
y = m.add(aicf.ReLU(name="relu1", save_for_bwd=False), y)
z = m.add(aicf.Add(name="res"), y, y)
m.output("z", z)

print(m.dump())
sanity(m)
