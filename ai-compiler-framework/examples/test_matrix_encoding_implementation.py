from __future__ import annotations

import numpy as np
from scipy import sparse
from typing import Any, List, Dict, Tuple
import sys
from pathlib import Path as _Path

# -----------------------
# project path bootstrap (구조에 맞춘 엄밀한 설정)
# -----------------------
p = _Path(__file__).resolve()
# pyproject.toml이 있는 최상위 루트를 찾음
root = next(parent for parent in [p] + list(p.parents) if (parent / "pyproject.toml").exists())

# C++ 바이너리 경로
build_lib_path = root / "build" / "python" / "aicf_cuda"
# Python 소스 경로
src_path = root / "python" / "aicf_v2" / "src"

if build_lib_path.exists():
    sys.path.insert(0, str(build_lib_path))
if src_path.exists():
    sys.path.insert(0, str(src_path))

import aicf_v2 as aicf

# -----------------------
# 1. Algebraic Graph Optimizer
# -----------------------
class AlgebraicGraphOptimizer:
    def __init__(self, builder: Any):
        self.b = builder
        self.ops = builder.ops
        self.n = len(self.ops)
        self.A = None
        self.S = {}
        
    def encode(self):
        """객체를 행렬로 인코딩"""
        # Vid -> Producer Op mapping
        val_to_producer = {vid: i for i, op in enumerate(self.ops) for vid in op.outputs}
        
        rows, cols = [], []
        for j, op in enumerate(self.ops):
            for v_in in op.inputs:
                if v_in in val_to_producer:
                    rows.append(val_to_producer[v_in])
                    cols.append(j)
        
        # 인접 행렬 A (CSR 포맷)
        self.A = sparse.csr_matrix(
            (np.ones(len(rows)), (rows, cols)), 
            shape=(self.n, self.n)
        )

        # Kind 선택 행렬 S
        unique_kids = set(getattr(op, "kind_id", -1) for op in self.ops)
        for kid in unique_kids:
            diag = [1 if getattr(op, "kind_id", -1) == kid else 0 for op in self.ops]
            self.S[kid] = sparse.diags(diag)
        
        print(f"[MAT-OPT] Encoded {self.n} ops with {len(rows)} edges.")


    def find_gbr_patterns(self) -> List[Tuple[int, int, int]]:
        """Gemm(2) -> BiasAdd(3) -> ReLU(1) 탐색"""
        # [수정] 행렬 객체가 None인지만 체크 (Truth value 모호성 해결)
        S_g = self.S.get(2)
        S_b = self.S.get(3)
        S_r = self.S.get(1)
        
        if S_g is None or S_b is None or S_r is None:
            print("[MAT-OPT] Required KIDs (2, 3, 1) not found in the graph.")
            return []

        # Algebraic Path: G -> B -> R
        # M_gb[i, j] = 1 이면 i(Gemm) -> j(Bias) 연결
        M_gb = S_g @ self.A @ S_b
        # M_br[j, k] = 1 이면 j(Bias) -> k(ReLU) 연결
        M_br = S_b @ self.A @ S_r
        
        # M_full[i, k] = 1 이면 i -> (j) -> k 경로 존재
        M_full = M_gb @ M_br
        
        results = []
        rows, cols = M_full.nonzero()
        for i, k in zip(rows, cols):
            # 중간 노드 j 식별 (i -> j -> k)
            # i행(Gemm의 출력들)과 k열(ReLU의 입력들)의 교집합인 j(Bias) 추출
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

    x_spec = aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda")
    _ = model.build(x_spec, input_name="x")

    opt = AlgebraicGraphOptimizer(model.b)
    opt.encode()
    
    patterns = opt.find_gbr_patterns()
    
    print("\n" + "="*60)
    print(f" [ALGEBRAIC MATCHING] FOUND {len(patterns)} TARGETS")
    print("="*60)
    for i, j, k in patterns:
        print(f"PATTERN: Op[{i}](Gemm) -> Op[{j}](Bias) -> Op[{k}](ReLU)")
    print("="*60)

    print("\n[ADJACENCY MATRIX A (TOPOLOGY)]")
    print(opt.A.toarray().astype(int))

if __name__ == "__main__":
    main()