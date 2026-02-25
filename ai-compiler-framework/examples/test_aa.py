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

# ---------------------------------------------------------
# 1. Fused Op Definition & Backward Logic
# ---------------------------------------------------------
class FusedGemmBiasReluOp:
    KIND = "fused_gemm_bias_relu"
    KID = 100 

    @staticmethod
    def emit_bwd(b: Any, ctx: Any, fused_node: Any, grad_y: int) -> Dict[int, int]:
        """
        Fused 노드가 역전파 시 호출받을 커스텀 미분 로직
        """
        print(f"\n[BWD-INVOKE] Fused node '{fused_node.name}' generating custom BWD ops...")
        
        # 입력 ID 추출: Gemm(0), Weight(1), Bias(2)
        x_vid, w_vid, b_vid = fused_node.inputs[0], fused_node.inputs[1], fused_node.inputs[2]
        
        # 그래디언트 값(가상) 생성 및 등록
        grads = {
            x_vid: b.value(f"{fused_node.name}.grad_x", b.values[x_vid].spec),
            w_vid: b.value(f"{fused_node.name}.grad_w", b.values[w_vid].spec),
            b_vid: b.value(f"{fused_node.name}.grad_b", b.values[b_vid].spec),
        }
        
        print(f" >>> Gradients registered for: x({x_vid}), w({w_vid}), bias({b_vid})")
        return grads

# ---------------------------------------------------------
# 2. Rich Algebraic Optimizer (Pattern Discovery)
# ---------------------------------------------------------
class RichMatrixOptimizer:
    FLAG_INPLACE_OK = 1 << 0
    FLAG_DTYPE_F32  = 1 << 1
    FLAG_SAFE_NODE  = 1 << 2 # Out-degree <= 1

    def __init__(self, builder: Any):
        self.b = builder
        self.ops = builder.ops
        self.n = len(self.ops)
        self.A_rich: sparse.csr_matrix = None
        self.S: Dict[int, sparse.dia_matrix] = {}

    def encode(self):
        val_to_producer = {vid: i for i, op in enumerate(self.ops) for vid in op.outputs}
        
        # Out-degree 계산
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
                    flag = 0
                    if getattr(op, "constraints", {}).get("inplace_ok"): flag |= self.FLAG_INPLACE_OK
                    if self.b.values[v_in].spec.dtype == "f32": flag |= self.FLAG_DTYPE_F32
                    if counts[i] <= 1: flag |= self.FLAG_SAFE_NODE
                    
                    rows.append(i)
                    cols.append(j)
                    data.append(flag)
        
        self.A_rich = sparse.csr_matrix((data, (rows, cols)), shape=(self.n, self.n))
        
        unique_kids = set(getattr(op, "kind_id", -1) for op in self.ops)
        for kid in unique_kids:
            diag = [1 if getattr(op, "kind_id", -1) == kid else 0 for op in self.ops]
            self.S[kid] = sparse.diags(diag)

    def find_fused_gemm_bias_relu(self) -> List[Tuple[int, int, int]]:
        S_g, S_b, S_r = self.S.get(2), self.S.get(3), self.S.get(1)
        if any(s is None for s in [S_g, S_b, S_r]): return []

        # 필터: F32이면서 분기되지 않는(Safe) 연결만 허용
        required_bits = self.FLAG_SAFE_NODE | self.FLAG_DTYPE_F32
        A_valid_data = (self.A_rich.data & required_bits) == required_bits
        A_valid = sparse.csr_matrix((A_valid_data.astype(int), self.A_rich.indices, self.A_rich.indptr), shape=(self.n, self.n))

        # 대수적 매칭 (Gemm -> Bias -> ReLU)
        M_gb = S_g @ A_valid @ S_b
        M_br = S_b @ A_valid @ S_r
        M_full = M_gb @ M_br
        
        results = []
        rows, cols = M_full.nonzero()
        for i, k in zip(rows, cols):
            j_indices = (M_gb[i, :].multiply(M_br[:, k].T)).nonzero()[1]
            if len(j_indices) > 0:
                results.append((int(i), int(j_indices[0]), int(k)))
        return results

# ---------------------------------------------------------
# 3. Graph Rewriter (Fusion Implementation)
# ---------------------------------------------------------
class GraphRewriter:
    def __init__(self, builder: Any):
        self.b = builder

    def apply_fusion(self, patterns: List[Tuple[int, int, int]]):
        for i, j, k in patterns:
            op_g, op_b, op_r = self.b.ops[i], self.b.ops[j], self.b.ops[k]
            print(f"[REWRITE] Fusing Ops: {i}(Gemm) -> {j}(Bias) -> {k}(ReLU)")

            # 1) 대표 노드(Gemm)를 Fused 노드로 변환
            op_g.kind = FusedGemmBiasReluOp.KIND
            op_g.kind_id = FusedGemmBiasReluOp.KID
            op_g.name = f"{op_g.name}_fused"
            # 입력 통합: [Gemm_X, Gemm_W, Bias_B]
            op_g.inputs = [op_g.inputs[0], op_g.inputs[1], op_b.inputs[1]]
            # 출력 통합: 최종 ReLU의 출력 사용
            op_g.outputs = op_r.outputs
            
            # 2) [핵심] Backward 생성을 위한 훅 주입
            setattr(op_g, 'bwd_emit_fn', FusedGemmBiasReluOp.emit_bwd)

            # 3) 나머지 노드는 NOP(No Operation) 처리
            for idx in [j, k]:
                self.b.ops[idx].kind = "nop"
                self.b.ops[idx].inputs, self.b.ops[idx].outputs = [], []


def print_rich_adjacency_details(opt: RichMatrixOptimizer):
    """연결 관계 행렬의 비트 플래그를 해석하여 출력"""
    print("\n[DEBUG] Rich Adjacency Matrix Details:")
    print("Row (Producer) -> Col (Consumer) | Flags (Safe:4, F32:2, Inplace:1)")
    print("-" * 65)
    
    # CSR 행렬에서 0이 아닌 요소(연결된 노드들) 추출
    A = opt.A_rich.tocoo() 
    for r, c, val in zip(A.row, A.col, A.data):
        flags = []
        if val & opt.FLAG_SAFE_NODE: flags.append("SAFE")
        if val & opt.FLAG_DTYPE_F32:  flags.append("F32")
        if val & opt.FLAG_INPLACE_OK: flags.append("INPLACE")
        
        flag_str = "|".join(flags) if flags else "NONE"
        print(f"Op[{r:02d}] --> Op[{c:02d}] | Raw Value: {val:d} | Interpreted: ({flag_str})")

# ---------------------------------------------------------
# 4. Integrated Execution
# ---------------------------------------------------------



def main():
    # 1. 모델 빌드
    model = aicf.Sequential([
        aicf.Linear(784, 128, name="fc1"),
        aicf.ReLU(name="relu1"),
    ])
    x_spec = aicf.TensorSpec(shape=(64, 784), dtype="f32", device="cuda")
    y_vid = model.build(x_spec, input_name="x")
    
    # Loss 추가 (그래프 구조 형성을 위함)
    from aicf_v2.emitters.cuda.reduce_sum import emit as emit_reduce
    final_loss_vid = model.b.value("final_loss", aicf.TensorSpec(shape=(1,), dtype="f32", device="cuda"))
    emit_reduce(model.b, model.ctx, x=y_vid, out=final_loss_vid, axis=0)

    # 2. Rich Matrix Optimizer 실행
    opt = RichMatrixOptimizer(model.b)
    opt.encode()

    print("\n" + "="*80)
    print(" [PHASE 1] Rich Matrix Connection Analysis")
    print("="*80)

    # 연결 관계 행렬 출력
    print("\n[Adjacency Matrix Visualization (Raw Flags)]")
    # 넘파이 출력 옵션을 조절하여 깔끔하게 표시
    print(np.array2string(opt.A_rich.toarray().astype(int), separator='  '))
    
    # 상세 비트 해석 출력
    print_rich_adjacency_details(opt)

    # 3. 패턴 매칭 및 리라이팅
    targets = opt.find_fused_gemm_bias_relu()
    rewriter = GraphRewriter(model.b)
    rewriter.apply_fusion(targets)

    # 4. Backward 생성
    print("\n" + "="*80)
    print(" [PHASE 2] Backward Generation")
    print("="*80)
    model.build_backward_after_fwd_opt(final_loss_vid)

    # 최종 상태 출력
    print("\n[FINAL IR SNAPSHOT]")
    for i, op in enumerate(model.b.ops):
        suffix = " <--- FWD OPTIMIZED" if op.kind == "fused_gemm_bias_relu" else ""
        print(f"Op[{i:02d}]: {op.kind:<25} {suffix} | Name: {op.name}")

if __name__ == "__main__":
    main()