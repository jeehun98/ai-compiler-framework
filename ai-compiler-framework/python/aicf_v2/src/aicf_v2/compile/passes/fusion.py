from __future__ import annotations
import numpy as np
from scipy import sparse
from typing import Any, List, Dict, Tuple, Optional

from aicf_v2.emitters.cuda.base import OpFlags
from aicf_v2.emitters.cuda import gemm_epilogue


# -----------------------------------------------------------------------------
# Declarative Fusion Registry (bit-sequence based)
# -----------------------------------------------------------------------------
FUSION_PATTERNS = {
    "gemm_epilogue": {
        "sequence": [
            OpFlags.IS_GEMM_LIKE,                             # Root (Gemm)
            OpFlags.IS_ELEMENTWISE | OpFlags.HAS_BIAS,        # BiasAdd-like
            OpFlags.IS_ELEMENTWISE | OpFlags.IS_ACTIVATION,   # Relu-like
        ],
        "target_kind": "gemm_epilogue",
    },
}


# -----------------------------------------------------------------------------
# Matcher: 그래프 위상과 비트 지문을 분석하여 패턴을 탐색
# -----------------------------------------------------------------------------
class RichMatrixOptimizer:
    def __init__(self, builder: Any):
        self.b = builder
        self.ops = builder.ops
        self.n = len(self.ops)
        self.node_masks = np.zeros(self.n, dtype=np.uint32)
        self.adj_csr = None

    def encode(self):
        """그래프 위상(CSR) + 노드 비트마스크 생성"""
        val_to_producer = {vid: i for i, op in enumerate(self.ops) for vid in op.outputs}

        # Out-degree 계산 -> SAFE_NODE(단일 소비자) 판별
        temp_rows = []
        for op in self.ops:
            for v_in in op.inputs:
                if v_in in val_to_producer:
                    temp_rows.append(val_to_producer[v_in])
        counts = np.bincount(temp_rows, minlength=self.n) if temp_rows else np.zeros(self.n)

        for i, op in enumerate(self.ops):
            mask = int(getattr(op, "static_flags", OpFlags.NONE))
            if counts[i] <= 1:
                mask |= OpFlags.SAFE_NODE
            self.node_masks[i] = mask

        # CSR adjacency 생성
        rows, cols = [], []
        for j, op in enumerate(self.ops):
            for v_in in op.inputs:
                if v_in in val_to_producer:
                    rows.append(val_to_producer[v_in])
                    cols.append(j)
        self.adj_csr = sparse.csr_matrix(
            (np.ones(len(rows), dtype=np.uint8), (rows, cols)),
            shape=(self.n, self.n),
        )

    def find_matches(self, pattern_name: str) -> List[List[int]]:
        """비트 시퀀스에 매칭되는 노드 인덱스 경로 반환"""
        config = FUSION_PATTERNS[pattern_name]
        seq = config["sequence"]

        # Root: 첫 번째 비트 조건이 충족되는 노드 탐색
        roots = np.where((self.node_masks & seq[0]) == seq[0])[0]

        results: List[List[int]] = []
        for r in roots:
            self._match_recursive([int(r)], seq[1:], results)
        return results

    def _match_recursive(self, path: List[int], remaining_seq: List[int], results: List[List[int]]):
        if not remaining_seq:
            results.append(list(path))
            return

        last = path[-1]
        need = remaining_seq[0]

        for c in self._get_consumers(last):
            c = int(c)
            # 비트마스크 일치 여부 확인
            if (self.node_masks[c] & need) != need:
                continue

            # 중간 노드는 안전을 위해 SAFE_NODE 제약 확인
            if len(remaining_seq) > 1 and not (self.node_masks[c] & OpFlags.SAFE_NODE):
                continue

            path.append(c)
            self._match_recursive(path, remaining_seq[1:], results)
            path.pop()

    def _get_consumers(self, idx: int) -> np.ndarray:
        return self.adj_csr.indices[self.adj_csr.indptr[idx] : self.adj_csr.indptr[idx + 1]]


# -----------------------------------------------------------------------------
# Rewriter: 탐색된 패턴을 실제 최적화된 노드로 치환
# -----------------------------------------------------------------------------
class GraphRewriter:
    def __init__(self, builder: Any, ctx: Any):
        self.b = builder
        self.ctx = ctx  # 주입받은 Context 보관 (Emitter 호출 시 사용)

    def apply_fusion(self, pattern_name: str, matches: List[List[int]]):
        """인덱스 안정성을 유지하며 퓨전 적용"""
        if pattern_name != "gemm_epilogue":
            return

        # 겹침 방지를 위해 루트 인덱스 역순으로 처리
        matches_sorted = sorted(matches, key=lambda p: p[0], reverse=True)
        for path in matches_sorted:
            self._fuse_gemm_epilogue(path)

    def _fuse_gemm_epilogue(self, path: List[int]):
        # path: [gemm, bias_add, relu]
        if len(path) != 3:
            return

        i_g, i_b, i_r = path
        op_g = self.b.ops[i_g]
        op_b = self.b.ops[i_b]
        op_r = self.b.ops[i_r]

        if getattr(op_g, "kind", "") == "nop": return

        # 1) 속성 및 입력 추출
        ta = bool(op_g.attrs.get("transA", False))
        tb = bool(op_g.attrs.get("transB", False))
        A_vid, B_vid = op_g.inputs[0], op_g.inputs[1]

        # Bias Vid 추출 (gemm 출력이 아닌 입력 찾기)
        prev_out = set(op_g.outputs)
        bias_vid = next((v for v in op_b.inputs if v not in prev_out), op_b.inputs[-1])
        out_vid = op_r.outputs[0]

        # 2) Fused Emitter 호출 (ctx 사용)
        # b.ops 리스트 끝에 새로운 노드가 추가됨
        new_op_idx = gemm_epilogue.emit(
            self.b, self.ctx,
            A=A_vid, B=B_vid, bias=bias_vid, out=out_vid,
            transA=ta, transB=tb, relu=True,
            name=f"{self.b.ops[i_g].name}_fused"
        )

        # 2. 생성된 노드 객체를 '복사'하여 가져오기
        import copy
        fused_op_orig = self.b.ops[new_op_idx]
        fused_op_copy = copy.copy(fused_op_orig) # 얕은 복사로 충분함
        
        # 3. 역전파 훅 주입
        setattr(fused_op_copy, "bwd_emit_fn", gemm_epilogue.emit_bwd)

        # 4. 기존 Gemm 위치[i_g]를 Fused Op로 대체
        self.b.ops[i_g] = fused_op_copy

        # 5. 나머지 노드들(기존 b, r 그리고 방금 생성에 사용된 꼬리 노드)을 nop화
        # 주의: i_g 자리에 이미 복사본을 넣었으므로 new_op_idx를 마음편히 nop으로 만듭니다.
        for idx in (i_b, i_r, new_op_idx):
            self.b.ops[idx].kind = "nop"
            self.b.ops[idx].inputs = []
            self.b.ops[idx].outputs = []
            self.b.ops[idx].attrs = {}


# -----------------------------------------------------------------------------
# Pipeline Entry: Context를 주입받아 최적화 공정 실행
# -----------------------------------------------------------------------------
def optimize_ir(b: Any, ctx: Any):
    # 1. 분석
    opt = RichMatrixOptimizer(b)
    opt.encode()

    # 2. 치환 (Context 주입)
    rewriter = GraphRewriter(b, ctx)
    for pname in FUSION_PATTERNS.keys():
        matches = opt.find_matches(pname)
        if matches:
            rewriter.apply_fusion(pname, matches)