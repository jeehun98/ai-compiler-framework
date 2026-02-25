from __future__ import annotations
import numpy as np
from scipy import sparse
from typing import Any, List, Dict, Tuple

class FusedGemmBiasReluOp:
    KIND = "fused_gemm_bias_relu"
    KID = 100 

    @staticmethod
    def emit_bwd(b: Any, ctx: Any, fused_node: Any, grad_y: int) -> Dict[int, int]:
        """Fused 노드의 역전파 로직"""
        # 입력 ID 추출: Gemm_X(0), Gemm_W(1), Bias_B(2)
        x_vid, w_vid, b_vid = fused_node.inputs[0], fused_node.inputs[1], fused_node.inputs[2]
        
        grads = {
            x_vid: b.value(f"{fused_node.name}.grad_x", b.values[x_vid].spec),
            w_vid: b.value(f"{fused_node.name}.grad_w", b.values[w_vid].spec),
            b_vid: b.value(f"{fused_node.name}.grad_b", b.values[b_vid].spec),
        }
        return grads

class RichMatrixOptimizer:
    FLAG_INPLACE_OK = 1 << 0
    FLAG_DTYPE_F32  = 1 << 1
    FLAG_SAFE_NODE  = 1 << 2

    def __init__(self, builder: Any):
        self.b = builder
        self.ops = builder.ops
        self.n = len(self.ops)
        self.A_rich = None
        self.S = {}

    def encode(self):
        val_to_producer = {vid: i for i, op in enumerate(self.ops) for vid in op.outputs}
        
        # Out-degree 계산 (분기 여부 판별)
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

        required_bits = self.FLAG_SAFE_NODE | self.FLAG_DTYPE_F32
        A_valid_data = (self.A_rich.data & required_bits) == required_bits
        A_valid = sparse.csr_matrix((A_valid_data.astype(int), self.A_rich.indices, self.A_rich.indptr), shape=(self.n, self.n))

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

class GraphRewriter:
    def __init__(self, builder: Any):
        self.b = builder

    def apply_fusion(self, patterns: List[Tuple[int, int, int]]):
        for i, j, k in patterns:
            op_g, op_b, op_r = self.b.ops[i], self.b.ops[j], self.b.ops[k]
            
            # Forward 통합
            op_g.kind = FusedGemmBiasReluOp.KIND
            op_g.kind_id = FusedGemmBiasReluOp.KID
            op_g.name = f"{op_g.name}_fused"
            op_g.inputs = [op_g.inputs[0], op_g.inputs[1], op_b.inputs[1]]
            op_g.outputs = op_r.outputs
            
            # Backward 훅 주입
            setattr(op_g, 'bwd_emit_fn', FusedGemmBiasReluOp.emit_bwd)

            # 중간 노드 무효화 (NOP)
            for idx in [j, k]:
                self.b.ops[idx].kind = "nop"
                self.b.ops[idx].inputs, self.b.ops[idx].outputs = [], []