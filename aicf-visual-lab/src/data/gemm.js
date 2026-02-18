export const gemmData = {
  id: "GEMM",
  category: "선형 변환 / 특징 투영 (Linear Projection)",

  canonical: {
    formula: "C = \\alpha(A \\times B) + \\beta C",
    shapes: { A: "M x K", B: "K x N", C: "M x N" },
    interpretation: {
      row_A: "샘플 i (입력 질의)",
      col_B: "가설 j (비교 특징)",
      c_ij: "샘플 i와 가설 j의 연관성 점수",
    },
  },

  semantics: {
    thesis: "관계 가설을 평가하고 이전 상태를 병합하는 의미론적 투영 연산자",
    axes: {
      M: { name: "샘플 (Samples)", role: "질의 배치 (Batch)" },
      K: { name: "가설 탐색 공간 (Hypothesis Space)", role: "의미론적 근거 축적 공간" },
      N: { name: "특징 채널 (Features)", role: "투영 결과값" },
    },

    invariants: [
      {
        id: "INV_TOPK_ORDER",
        name: "순위 보존성 (Rank Invariance)",
        metric: "행 단위 순서(argsort) 일치도",
        threshold: "상위 K개 결과의 순서 유지",
        allows: ["저정밀도 양자화", "근사 행렬곱", "조기 종료"],
      },
      {
        id: "INV_BOUNDARY",
        name: "결정 경계 보존성 (Decision Boundary)",
        metric: "부호 일치성 (Sign Consistency)",
        threshold: "99.99% 이상 일치",
        allows: ["검증된 범위 내의 과감한 연산 재작성"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "ReLU 기반 조기 종료", // 명확한 액션 중심
          rule: "음수 확정 영역(C << 0) 감지 시, 정밀도 포기 및 연산 즉시 중단",
          hint: "불필요한 음수 정밀도 제거",
        },
        {
          name: "Softmax 기반 데이터 가지치기",
          rule: "최댓값과 격차가 큰 하위 확률 요소는 지수 계산(Exp) 전 생략",
          hint: "무의미한 하위 확률 연산 생략",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "TensorCore_GEMM_EpilogueFused",
      reason: [
        "후속 Softmax 연산 감지: 순위 보존 불변성 강제 적용",
        "프로파일링 결과: 타일의 78%에서 조기 종료 안전 확인 (성능 최적화)",
        "에필로그 상태 병합 활성화: 중간 데이터 생성 없이 즉시 융합",
      ],
      applied_rewrites: ["연산 앵커 결합 (Anchor Fusion)", "누적 연산 조기 종료 (Early Exit)"],
    },
  },

  kernel: {
    strategy: "2단계 계층적 타일링 (스트라이드 접근 최적화)",
    details: [
      { technique: "공유 메모리 타일링", semantic_link: "K축(가설 공간) 데이터 재사용 극대화" },
      { technique: "K-루프 언롤링", semantic_link: "가설 검증 처리량 가속" },
      { technique: "에필로그 융합", semantic_link: "상태 병합은 하나의 의미 단위이므로 쓰기 지연 방지" },
    ],
    metrics: { 
      memory_reuse: "14.2배", 
      throughput: "3,188.9 GF/s (실측치)", 
      occupancy: "92%" 
    },
  },
};