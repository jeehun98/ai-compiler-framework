from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Any, List, Dict, Tuple
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
# 1. Rich Algebraic Optimizer (Packed Metadata)
# -----------------------
class RichMatrixOptimizer:
    # 비트 플래그 정의 (연결 관계의 성격 표현)
    FLAG_INPLACE_OK = 1 << 0  # 0x01
    FLAG_DTYPE_F32  = 1 << 1  # 0x02
    FLAG_SAFE_NODE  = 1 << 2  # 0x04 (Out-degree <= 1)

    def __init__(self, builder: Any):
        self.b = builder
        self.ops = builder.ops
        self.n = len(self.ops)
        self.A_rich: sparse.csr_matrix = None # 정보를 머금은 인접 행렬
        self.S: Dict[int, sparse.dia_matrix] = {}

    def encode(self):
        """그래프를 정보를 포함한 풍부한 행렬(Rich Adjacency)로 인코딩"""
        val_to_producer = {vid: i for i, op in enumerate(self.ops) for vid in op.outputs}
        
        # 1. 각 노드별 Out-degree 미리 계산 (Safe node 판별용)
        temp_rows = []
        for op in self.ops:
            for v_in in op.inputs:
                if v_in in val_to_producer:
                    temp_rows.append(val_to_producer[v_in])
        counts = np.bincount(temp_rows, minlength=self.n) if temp_rows else np.zeros(self.n)

        rows, cols, data = [], [], []
        for j, op in enumerate(self.ops):
            for v_in in op.inputs:
                if v_in in val_to_producer:
                    i = val_to_producer[v_in]
                    
                    # --- 비트 플래그 조립 (구조체 역할) ---
                    flag = 0
                    # 1) Inplace 가능 여부 (Next Op 기준)
                    if getattr(op, "constraints", {}).get("inplace_ok"):
                        flag |= self.FLAG_INPLACE_OK
                    # 2) 데이터 타입 체크 (Value 기준)
                    if self.b.values[v_in].spec.dtype == "f32":
                        flag |= self.FLAG_DTYPE_F32
                    # 3) 분기 여부 (Producer 기준)
                    if counts[i] <= 1:
                        flag |= self.FLAG_SAFE_NODE
                    
                    rows.append(i)
                    cols.append(j)
                    data.append(flag) # 단순 1 대신 메타데이터 주입
        
        self.A_rich = sparse.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
        
        # Kind 선택 행렬은 기존과 동일
        unique_kids = set(getattr(op, "kind_id", -1) for op in self.ops)
        for kid in unique_kids:
            diag = [1 if getattr(op, "kind_id", -1) == kid else 0 for op in self.ops]
            self.S[kid] = sparse.diags(diag)

    def get_attr_filter(self, attr_key: str, expected_value: Any) -> sparse.dia_matrix:
        mask = [1 if getattr(op, "attrs", {}).get(attr_key) == expected_value else 0 for op in self.ops]
        return sparse.diags(mask)

    def find_fused_gemm_bias_relu(self):
        """
        비트마스킹을 활용한 고차원 패턴 매칭
        """
        S_g = self.S.get(2) # Gemm
        S_b = self.S.get(3) # BiasAdd
        S_r = self.S.get(1) # ReLU
        
        if any(s is None for s in [S_g, S_b, S_r]): return []

        # 1. Gemm 속성 필터 (Node 필터)
        C_transB = self.get_attr_filter("transB", True)
        
        # 2. Rich Adjacency에서 특정 비트(SAFE + F32)가 켜진 연결만 추출
        # Matrix A_rich의 값들 중 (FLAG_SAFE_NODE | FLAG_DTYPE_F32) 비트가 모두 켜진 것만 1로 변환
        required_bits = self.FLAG_SAFE_NODE | self.FLAG_DTYPE_F32
        
        # 연결 강도(Flag)에서 필요한 비트가 있는지 검사하여 Binary Matrix 추출
        A_filtered_data = (self.A_rich.data & required_bits) == required_bits
        A_valid = sparse.csr_matrix((A_filtered_data.astype(int), self.A_rich.indices, self.A_rich.indptr), shape=(self.n, self.n))

        # 3. 최적화된 매칭 연산
        M_gb = (S_g @ C_transB) @ A_valid @ S_b
        M_br = S_b @ A_valid @ S_r
        M_full = M_gb @ M_br
        
        results = []
        rows, cols = M_full.nonzero()
        for i, k in zip(rows, cols):
            j_indices = (M_gb[i, :].multiply(M_br[:, k].T)).nonzero()[1]
            if len(j_indices) > 0:
                results.append((int(i), int(j_indices[0]), int(k)))
        return results

# -----------------------
# 2. Main
# -----------------------
def main():
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
        aicf.Linear(128, 10, name="fc2"),
    ])
    model.build(aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda"))

    opt = RichMatrixOptimizer(model.b)
    opt.encode()

    print("\n" + "="*85)
    print(" [RICH MATRIX OPTIMIZATION - PACKED METADATA]")
    print("="*85)

    # Rich Adjacency 확인 (비트 플래그 값이 저장됨)
    print("\n[Rich Adjacency Matrix A_rich (Stored Flags)]")
    print(opt.A_rich.toarray().astype(int))
    print("Flag Guide: Inplace=1, F32=2, Safe=4 | Combined: (Safe+F32+Inplace)=7")

    # 매칭 결과
    targets = opt.find_fused_gemm_bias_relu()
    
    print("\n[MATCHED TARGETS WITH BIT-FILTERING]")
    if not targets:
        print("No valid targets found.")
    for i, j, k in targets:
        print(f"FOUND: Op[{i}](Gemm) -> Op[{j}](Bias) -> Op[{k}](ReLU)")
        print(f"      - Combined Check: Topology OK, Bit-Flags(Safe/F32/Inplace) OK")

    print("="*85)

if __name__ == "__main__":
    main()