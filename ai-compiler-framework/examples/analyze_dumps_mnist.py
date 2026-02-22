from __future__ import annotations
import json
from pathlib import Path
from collections import Counter, defaultdict

DUMP_DIR = Path("./aicf_dumps_mnist")

def load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))

def summarize_builder(builder_json: dict, title: str):
    ops = builder_json.get("ops", [])
    values = builder_json.get("values", {})
    out_map = builder_json.get("output_map", {})

    kinds = [op.get("kind") for op in ops]
    c = Counter(kinds)

    print("\n" + "="*80)
    print(f"[BUILDER] {title}")
    print("="*80)
    print(f"num_ops={len(ops)}  num_values={builder_json.get('num_values')}")
    print(f"inputs={len(builder_json.get('input_vids', []))}  params={len(builder_json.get('param_vids', []))}  states={len(builder_json.get('state_vids', []))}")
    print(f"externals={len(builder_json.get('external_vids', []))}  outputs={out_map}")

    print("\nTop op kinds:")
    for k, v in c.most_common(20):
        print(f"  {k}: {v}")

    # 간단한 패턴 체크: grad_acc_* 생성 개수
    grad_acc = [v for v in values.values() if isinstance(v, dict) and str(v.get("name","")).startswith("grad_acc_")]
    print(f"\ngrad_acc_* values: {len(grad_acc)}")

def summarize_plan(plan_json: dict, title: str):
    alias = plan_json.get("alias", {})
    ops = plan_json.get("ops", [])
    kinds = [op.get("kind") for op in ops]
    c = Counter(kinds)

    print("\n" + "="*80)
    print(f"[PLAN] {title}")
    print("="*80)
    print(f"plan_id={plan_json.get('plan_id')}")
    print(f"num_ops={len(ops)}  alias_pairs={len(alias)}")

    print("\nTop op kinds:")
    for k, v in c.most_common(30):
        print(f"  {k}: {v}")

    # kind_id 누락 체크
    missing_kind_id = [op for op in ops if op.get("kind_id") is None]
    if missing_kind_id:
        print(f"\n[WARN] missing kind_id ops: {len(missing_kind_id)}")
    else:
        print("\nkind_id: all present ✅")

    # attr_schema/attr_blob_len 체크
    missing_attr = [op for op in ops if op.get("attr_schema") is None]
    print(f"attr_schema missing: {len(missing_attr)}")

    # alias 샘플 몇 개
    if alias:
        print("\nAlias samples (first 10):")
        for i, (k, v) in enumerate(list(alias.items())[:10]):
            print(f"  {k} -> {v}")

def main():
    fwd = load(DUMP_DIR / "01_fwd_built__builder.json")
    bwd = load(DUMP_DIR / "02_bwd_built__builder.json")
    plan = load(DUMP_DIR / "03_compiled_plan__plan.json")

    summarize_builder(fwd, "01_fwd_built")
    summarize_builder(bwd, "02_bwd_built")
    summarize_plan(plan, "03_compiled_plan")

    # 추가: fwd vs bwd op kind 변화 비교
    fwd_k = Counter([op.get("kind") for op in fwd.get("ops", [])])
    bwd_k = Counter([op.get("kind") for op in bwd.get("ops", [])])

    print("\n" + "="*80)
    print("[DIFF] fwd -> bwd (op kind delta)")
    print("="*80)
    all_kinds = sorted(set(fwd_k) | set(bwd_k))
    for k in all_kinds:
        d = bwd_k[k] - fwd_k[k]
        if d != 0:
            sign = "+" if d > 0 else ""
            print(f"  {k}: {fwd_k[k]} -> {bwd_k[k]}  ({sign}{d})")

if __name__ == "__main__":
    main()