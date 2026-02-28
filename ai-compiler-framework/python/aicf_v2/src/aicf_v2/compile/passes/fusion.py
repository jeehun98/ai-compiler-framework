from __future__ import annotations
import numpy as np
from scipy import sparse
from typing import Any, List, Dict, Tuple, Optional

# Emitter의 base.py에 정의된 비트마스크 클래스를 사용
from aicf_v2.emitters.cuda.base import OpFlags

class FusedGemmBiasReluOp:
    """퓨전된 노드에 주입될 정체성 및 역전파 정의"""
    KIND = "fused_gemm_bias_relu"
    KID = 100 

    @staticmethod
    def emit_bwd(b: Any, ctx: Any, fused_node: Any, grad_y: int) -> Dict[int, int]:
        """Fused 노드의 역전파 로직: 입력 [X, W, Bias]에 대한 미분값 생성"""
        x_vid = fused_node.inputs[0]
        w_vid = fused_node.inputs[1]
        b_vid = fused_node.inputs[2]
        
        # 실제 런타임에서는 FusedBwd 커널이 실행되도록 IR을 구성
        grads = {
            x_vid: b.value(f"{fused_node.name}.dx", b.values[x_vid].spec),
            w_vid: b.value(f"{fused_node.name}.dw", b.values[w_vid].spec),
            b_vid: b.value(f"{fused_node.name}.db", b.values[b_vid].spec),
        }
        return grads

class RichMatrixOptimizer:
    def __init__(self, builder: Any):
        self.b = builder
        self.ops = builder.ops
        self.n = len(self.ops)
        
        # 탐색 가속을 위한 캐시 레이어
        self.node_masks = np.zeros(self.n, dtype=np.uint32)
        self.adj_csr = None

    def encode(self):
        """
        [Step 1] 정적 속성(Static)과 동적 속성(Derived)을 비트셋으로 통합
        [Step 2] CSR(Compressed Sparse Row)로 그래프 위상 정보 캡슐화
        """
        # Value ID -> 생산자 Node Index 매핑
        val_to_producer = {vid: i for i, op in enumerate(self.ops) for vid in op.outputs}
        
        # 1. Out-degree 계산 (Consumer가 여럿이면 퓨전 시 데이터 유실 위험이 있어 필터링 필요)
        temp_rows = []
        for op in self.ops:
            for v_in in op.inputs:
                if v_in in val_to_producer:
                    temp_rows.append(val_to_producer[v_in])
        counts = np.bincount(temp_rows, minlength=self.n) if temp_rows else np.zeros(self.n)

        # 2. 통합 Node Mask 생성
        for i, op in enumerate(self.ops):
            # Emitter 단계에서 설정된 본질 정보 (IS_GEMM, IS_ELEMENTWISE 등)
            mask = getattr(op, "static_flags", OpFlags.NONE)
            
            # 동적 맥락 정보: 단일 소비자(Safe) 여부
            if counts[i] <= 1:
                mask |= OpFlags.SAFE_NODE
            
            # 데이터 타입 힌트 (F32일 때만 최적화 커널이 존재할 경우)
            if any(self.b.values[v].spec.dtype == "f32" for v in op.outputs):
                mask |= OpFlags.DTYPE_F32
                
            self.node_masks[i] = mask

        # 3. CSR 인접 행렬 생성 (local traversal 가속용)
        rows, cols = [], []
        for j, op in enumerate(self.ops):
            for v_in in op.inputs:
                if v_in in val_to_producer:
                    rows.append(val_to_producer[v_in])
                    cols.append(j)
        
        # 데이터는 존재 여부(1)만 마킹
        self.adj_csr = sparse.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(self.n, self.n))

    def find_gemm_bias_relu(self) -> List[Tuple[int, int, int]]:
        """
        패턴: Root(GEMM) -> [Safe] Consumer(BIAS_ADD) -> Consumer(RELU)
        """
        from aicf_v2.emitters.cuda.base import OpFlags
        results = []
        
        # 1. Root 후보 추출 (static_flags & 0b1)
        seeds = np.where(self.node_masks & OpFlags.IS_GEMM_LIKE)[0]

        for i in seeds:
            for j in self._get_consumers(i):
                # 2. 중간 노드 체크: IS_ELEMENTWISE 이면서 SAFE_NODE(단일 소비자) 인가?
                if not (self.node_masks[j] & OpFlags.IS_ELEMENTWISE): continue
                if not (self.node_masks[j] & OpFlags.SAFE_NODE): continue
                
                for k in self._get_consumers(j):
                    # 3. 마지막 노드 체크: IS_ELEMENTWISE 인가?
                    if not (self.node_masks[k] & OpFlags.IS_ELEMENTWISE): continue
                    
                    # 4. 정밀 검증: 로그에서 확인된 'bias_add' 명칭으로 매칭 (핵심 수정)
                    # kinds = ["gemm", "bias_add", "relu"] 로 전달
                    if self._verify_pattern(i, j, k, ["gemm", "bias_add", "relu"]):
                        results.append((int(i), int(j), int(k)))
        return results

    def _verify_pattern(self, i, j, k, kinds: List[str]) -> bool:
        """비트 필터링 통과 후 최종 명칭 확인"""
        # 로그 결과: Op[00]=gemm, Op[01]=bias_add, Op[02]=relu
        return (self.ops[i].kind == kinds[0] and 
                self.ops[j].kind == kinds[1] and 
                self.ops[k].kind == kinds[2])

    def _get_consumers(self, node_idx: int) -> np.ndarray:
        """CSR 포인터를 직접 사용하여 특정 노드의 자식 노드 인덱스 슬라이싱"""
        return self.adj_csr.indices[self.adj_csr.indptr[node_idx] : self.adj_csr.indptr[node_idx + 1]]

class GraphRewriter:
    def __init__(self, builder: Any):
        self.b = builder

    def apply_fusion(self, patterns: List[Tuple[int, int, int]]):
        """탐색된 패턴을 바탕으로 실제 그래프 구조를 변형 및 무효화(NOP)"""
        for i, j, k in patterns:
            op_g, op_b, op_r = self.b.ops[i], self.b.ops[j], self.b.ops[k]
            
            # Role 기반 인덱스 룩업: Bias 연산의 입력 중 Bias 텐서(c)를 찾음
            in_role_b = op_b.attrs.get("in_role", ["a", "c"])
            try:
                # 'add' emitter에서 정의한 역할명을 사용하여 정확한 인덱스 추출
                bias_idx = list(in_role_b).index("c") 
            except ValueError:
                bias_idx = 1
            bias_vid = op_b.inputs[bias_idx]
            
            # 1. Forward 통합: 시작 노드인 Gemm을 퓨전 노드로 갱신
            op_g.kind = FusedGemmBiasReluOp.KIND
            op_g.kind_id = FusedGemmBiasReluOp.KID
            op_g.name = f"{op_g.name}_fused"
            
            # 입력 재구성: [X, W, Bias]
            op_g.inputs = [op_g.inputs[0], op_g.inputs[1], bias_vid]
            # 출력 전이: 최종 노드인 Relu의 출력을 승계
            op_g.outputs = op_r.outputs
            
            # 2. Backward Logic 주입 (Autograd 시 호출됨)
            setattr(op_g, 'bwd_emit_fn', FusedGemmBiasReluOp.emit_bwd)

            # 3. 중간 노드(Add, Relu) 무효화
            # IR 배열의 순서를 깨지 않기 위해 'nop'으로 변경 처리
            for idx in [j, k]:
                self.b.ops[idx].kind = "nop"
                self.b.ops[idx].inputs, self.b.ops[idx].outputs = [], []