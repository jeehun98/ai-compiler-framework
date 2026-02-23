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
# 1. Advanced Algebraic Optimizer
# -----------------------
class AdvancedMatrixOptimizer:
    def __init__(self, builder: Any):
        self.b = builder
        self.ops = builder.ops
        self.n = len(self.ops)
        self.A: sparse.csr_matrix = None
        self.S: Dict[int, sparse.dia_matrix] = {}

    def encode(self):
        """그래프를 행렬(Topology A, Kind S)로 인코딩"""
        val_to_producer = {vid: i for i, op in enumerate(self.ops) for vid in op.outputs}
        rows, cols = [], []
        for j, op in enumerate(self.ops):
            for v_in in op.inputs:
                if v_in in val_to_producer:
                    rows.append(val_to_producer[v_in])
                    cols.append(j)
        
        self.A = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n, self.n))
        unique_kids = set(getattr(op, "kind_id", -1) for op in self.ops)
        for kid in unique_kids:
            diag = [1 if getattr(op, "kind_id", -1) == kid else 0 for op in self.ops]
            self.S[kid] = sparse.diags(diag)

    def get_attr_filter(self, attr_key: str, expected_value: Any) -> sparse.dia_matrix:
        """특정 속성값을 가진 노드들만 남기는 대각 행렬 C_attr 생성"""
        mask = []
        for op in self.ops:
            val = getattr(op, "attrs", {}).get(attr_key)
            mask.append(1 if val == expected_value else 0)
        return sparse.diags(mask)

    def get_safe_fusion_filter(self) -> sparse.dia_matrix:
        """Out-degree가 1 이하인 노드(분기 없음)만 남기는 대각 행렬 C_safe 생성"""
        # A 행렬의 행 방향 합(Row Sum)이 Out-degree
        out_degrees = np.array(self.A.sum(axis=1)).flatten()
        mask = [1 if deg <= 1 else 0 for deg in out_degrees]
        return sparse.diags(mask)

    def find_fused_gemm_bias_relu(self):
        """
        [Recipe] Gemm(transB=True) -> BiasAdd(safe) -> ReLU
        Formula: M = (S_g @ C_transB) @ A @ (S_b @ C_safe) @ A @ S_r
        """
        S_g = self.S.get(2) # Gemm
        S_b = self.S.get(3) # BiasAdd
        S_r = self.S.get(1) # ReLU
        
        if any(s is None for s in [S_g, S_b, S_r]): return []

        # 1. 속성 필터 (transB가 True여야 퓨전 커널 사용 가능하다고 가정)
        C_transB = self.get_attr_filter("transB", True)
        
        # 2. 안전 필터 (중간 노드 BiasAdd가 다른 곳에 데이터를 주지 않아야 함)
        C_safe = self.get_safe_fusion_filter()

        # 3. 수학적 필터링 조립
        # Gemm(with transB) -> BiasAdd(with No branch)
        M_gb = (S_g @ C_transB) @ self.A @ (S_b @ C_safe)
        # BiasAdd -> ReLU
        M_br = S_b @ self.A @ S_r
        
        # 최종 경로
        M_full = M_gb @ M_br
        
        results = []
        rows, cols = M_full.nonzero()
        for i, k in zip(rows, cols):
            j_idx = (M_gb[i, :].multiply(M_br[:, k].T)).nonzero()[1]
            if len(j_indices := j_idx) > 0:
                results.append((int(i), int(j_indices[0]), int(k)))
        return results

# -----------------------
# 2. Main Test Loop
# -----------------------
def main():
    # 그래프 빌드
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
        aicf.Linear(128, 10, name="fc2"),
    ])
    x_spec = aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda")
    model.build(x_spec, input_name="x")

    # 최적화 엔진 가동
    optimizer = AdvancedMatrixOptimizer(model.b)
    optimizer.encode()

    print("\n" + "="*80)
    print(" [ADVANCED MATRIX OPTIMIZATION ANALYSIS]")
    print("="*80)

    # 1. Safe Fusion 필터 확인 (Out-degree 분석)
    C_safe = optimizer.get_safe_fusion_filter()
    print(f"Safe Fusion Mask (1=Safe, 0=Branching): {C_safe.diagonal().astype(int)}")

    # 2. Attr 필터 확인 (transB 분석)
    C_transB = optimizer.get_attr_filter("transB", True)
    print(f"transB=True Mask (1=Match, 0=Mismatch): {C_transB.diagonal().astype(int)}")

    # 3. 최종 매칭
    targets = optimizer.find_fused_gemm_bias_relu()
    
    print("\n[MATCHED TARGETS]")
    if not targets:
        print("No valid fusion targets found.")
    for i, j, k in targets:
        print(f"FOUND: Op[{i}](Gemm) -> Op[{j}](BiasAdd) -> Op[{k}](ReLU)")
        print(f"      - Criteria: transB=True (OK), Out-degree=1 (OK)")

    print("\n[ADJACENCY MATRIX A]")
    print(optimizer.A.toarray().astype(int))
    print("="*80)

if __name__ == "__main__":
    main()