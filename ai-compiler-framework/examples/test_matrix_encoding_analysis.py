from __future__ import annotations

import struct
from typing import Any, Dict, List
import torch
import sys
from pathlib import Path as _Path

# -----------------------
# project path bootstrap
# -----------------------
p = _Path(__file__).resolve()
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())
build_lib_path = root / "build" / "python" / "aicf_cuda"
src_path = root / "python" / "aicf_v2" / "src"
if build_lib_path.exists():
    sys.path.insert(0, str(build_lib_path))
sys.path.insert(0, str(src_path))

import aicf_v2 as aicf

# -----------------------
# 1. Deep Builder Inspector
# -----------------------
def deep_inspect_builder(b: Any):
    print("\n" + "█"*120)
    print(f" [DEEP BUILDER INSPECTION] TOTAL_OPS: {len(b.ops)} | TOTAL_VALUES: {len(b.values)}")
    print("█"*120)

    # (A) Value 객체 분석: 행렬의 Edge 속성(Shape/Dtype) 파악용
    print("\n[PART 1: VALUE OBJECTS - Edge Properties]")
    print(f"{'Vid':<5} | {'Name':<20} | {'Shape':<20} | {'Dtype':<10} | {'Producer':<10} | {'Consumers'}")
    print("-" * 120)
    
    # 역방향 인덱싱 준비
    val_to_producer = {vid: i for i, op in enumerate(b.ops) for vid in op.outputs}
    val_to_consumers = {vid: [] for vid in range(len(b.values))}
    for i, op in enumerate(b.ops):
        for v_in in op.inputs:
            val_to_consumers[v_in].append(i)

    for vid, v in enumerate(b.values):
        if v is None: continue
        spec = getattr(v, "spec", None)
        shape = str(list(spec.shape)) if spec else "N/A"
        dtype = str(spec.dtype) if spec else "N/A"
        name = getattr(v, "name", f"v{vid}")
        producer = val_to_producer.get(vid, "EXTERNAL")
        consumers = val_to_consumers.get(vid, [])
        
        # Out-degree가 1보다 크면 행렬 최적화 시 '분기(Branch)'로 마킹해야 함
        branch_mark = " [BRANCH]" if len(consumers) > 1 else ""
        print(f"{vid:<5} | {name[:20]:<20} | {shape:<20} | {dtype:<10} | {str(producer):<10} | {str(consumers)}{branch_mark}")

    # (B) Op 객체 분석: 행렬의 Node 및 Attr 필터링용
    print("\n[PART 2: OP OBJECTS - Node & Attribute Specs]")
    print(f"{'Idx':<4} | {'Kind':<13} | {'KID':<4} | {'Inputs':<12} | {'Outputs':<12} | {'AttrBlob(Hex)'}")
    print("-" * 120)

    for i, op in enumerate(b.ops):
        kind = getattr(op, "kind", "N/A")
        kid = getattr(op, "kind_id", -1)
        inputs = getattr(op, "inputs", [])
        outputs = getattr(op, "outputs", [])
        blob = getattr(op, "attr_blob", b"")
        blob_hex = blob.hex(' ') if blob else "empty"
        
        # 상세 속성 및 메타데이터 추출
        attrs = getattr(op, "attrs", {})
        constraints = getattr(op, "constraints", {})
        hints = getattr(op, "hints", {})
        saved = getattr(op, "saved", [])

        print(f"{i:<4} | {kind:<13} | {kid:<4} | {str(inputs):<12} | {str(outputs):<12} | {blob_hex}")
        
        # 행렬화 시 '제약 조건'으로 활용될 데이터들
        detail_str = []
        if attrs: detail_str.append(f"ATTRS: {attrs}")
        if constraints: detail_str.append(f"CONST: {constraints}")
        if hints: detail_str.append(f"HINTS: {hints}")
        if saved: detail_str.append(f"SAVED_VIDS: {saved}")
        
        for line in detail_str:
            print(f"     └─ {line}")

    print("█"*120 + "\n")

# -----------------------
# 2. Main Test Scenario
# -----------------------
def main():
    device = "cuda"
    batch_size = 64
    
    # 모델 정의
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
        aicf.Linear(128, 10, name="fc2"),
    ])

    # FWD 그래프 생성
    x_spec = aicf.TensorSpec(shape=(batch_size, 784), dtype="f32", device="cuda")
    y_pred_vid = model.build(x_spec, input_name="x")
    y_true_vid = model.input("y_true", aicf.TensorSpec(shape=(batch_size,), dtype="i32", device="cuda"))
    loss_vid = model.add(aicf.CrossEntropyLoss(reduction="mean", name="loss"), y_pred_vid, y_true_vid)

    # Output 매핑
    model.b.outputs["loss"] = loss_vid
    model.b.outputs["prob"] = y_pred_vid

    # [핵심 분석] FWD 상태 심층 해부
    deep_inspect_builder(model.b)

if __name__ == "__main__":
    main()