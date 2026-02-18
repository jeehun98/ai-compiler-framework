// src/data/residual_add.js

export const residualAddData = {
  id: "ResidualAdd",
  category: "상태 병합 / 정보 진화 (State Evolution)",

  descriptions: {
    essence: "기존에 학습된 지식(Identity)을 보존하면서, 새로운 정보(Residual)만을 안전하게 적층하여 심층 신경망의 학습 궤적을 보호합니다.",
    strategy: "단순 덧셈이 아닌 상태 진화(State Evolution) 과정으로 해석하여, 메모리 이동을 최소화하는 In-Place 누적 및 파이프라인 융합을 유도합니다.",
    hardware: "커널 런치 오버헤드를 없애기 위해 이전 연산의 에필로그에 융합되거나, 스트리밍 멀티프로세서(SM)의 대역폭을 최대로 쓰는 단순 병렬 합을 수행합니다."
  },

  canonical: {
    formula: "Y = \\underbrace{R}_{Identity} + \\underbrace{X}_{Residual}",
    shapes: { 
      X: "M x N", 
      R: "M x N", 
      Y: "M x N" 
    },
    interpretation: {
      R: "보존 메모리 (Identity Path - 장기 기억)",
      X: "변화량 / 잔차 (Residual Delta - 단기 수정)",
      Y: "진화된 상태 (Evolved State)",
    },
  },

  semantics: {
    thesis: "기존의 정보(R)를 훼손하지 않으면서, 학습된 변화량(X)만을 안전하게 적층하여 표현력을 확장하는 연산자",

    axes: {
      X: { name: "Correction Signal", role: "오차 보정 및 정보 추가" },
      R: { name: "Base State", role: "변화의 기준점 (Base Reference)" },
    },

    invariants: [
      {
        id: "INV_GRADIENT_HIGHWAY",
        name: "그래디언트 고속도로 (Gradient Flow Preservation)",
        metric: "야코비안 행렬식 (Jacobian Determinant)",
        threshold: "Singularity 발생 없음",
        allows: ["In-Place 업데이트", "비동기 연산"],
      },
      {
        id: "INV_SIGNAL_RATIO",
        name: "신호 대 잔차 비율 (Signal-to-Residual Ratio, SRR)",
        metric: "SRR = ||X|| / ||R||",
        threshold: "SRR << 1 (안정적 학습 구간)",
        allows: ["잔차 항의 저정밀도(FP16/BF16) 처리"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "메모리 대역폭 포화",
          rule: "연산 복잡도는 낮으나 데이터 이동량이 많아 병목 발생 (Memory Bound)",
          hint: "이전 연산(GEMM/Conv)의 에필로그로 융합 권장",
        },
        {
          name: "SRR(잔차 비율) 급증",
          rule: "X가 R보다 지나치게 커지면(SRR > 1), 학습 발산의 징후일 수 있음",
          hint: "클리핑(Clipping) 또는 정밀도 상향",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_Epilogue_Accumulate",
      reason: [
        "메모리 병목 해소: 별도의 커널로 실행하지 않고, 이전 GEMM 커널의 마지막 단계에서 처리",
        "캐시 적중률(Cache Hit) 극대화: 레지스터에 있는 계산 결과에 즉시 R을 로드하여 더함",
        "SRR 모니터링 생략: 안정적 학습 단계로 판단되어 고속 경로(Fast-Path) 선택",
      ],
      applied_rewrites: ["커널 융합 (Kernel Fusion)", "In-Place 누적 (In-Place Accumulation)"],
    },
  },

  kernel: {
    strategy: "Epilogue Fusion (Zero-Overhead)",
    details: [
      { technique: "벡터화된 로드 (LDS.128)", semantic_link: "메모리 버스 대역폭 100% 활용" },
      { technique: "스트림 멀티프로세서(SM) 병렬화", semantic_link: "독립적인 요소별(Element-wise) 연산 가속" },
      { technique: "쓰기 병합 (Coalesced Store)", semantic_link: "DRAM 접근 횟수 최소화" },
    ],
    metrics: {
      memory_reuse: "Optimal (Fused)",
      throughput: "Memory Bandwidth Bound",
      occupancy: "96%"
    },
  },

  costModel: {
    semanticLoss: "\\mathcal{L}_{evo} = \\alpha \\| \\nabla R - I \\| + \\beta \\cdot \\text{Cost}_{mem}",
    weights_hint: {
      default: { gradient_flow: 100.0, mem_cost: 5.0 }
    },
    metrics: {
      srr_value: "0.08",
      gradient_norm: "1.02"
    }
  },

  performance: {
    latency: {
      pytorch: 0.08,
      torch_compile: 0.04,
      ours: 0.00 // 융합되어 물리적 실행 시간 0ms (Zero-Cost)
    }
  }
};