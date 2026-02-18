// src/data/bias_add.js

export const biasAddData = {
  id: "BiasAdd",
  category: "상태 보정 / 경계 이동 (State Calibration)",

  descriptions: {
    essence: "활성화 함수 진입 전, 데이터 분포의 중심(Centroid)을 이동시켜 결정 경계의 영점을 보정하는 필수적인 상태 조정 단계입니다.",
    strategy: "단독 실행 대신 선행 연산(GEMM)의 에필로그나 후행 연산의 일부로 융합되어, 물리적인 커널 실행 자체를 제거하는 Zero-Cost 전략을 수행합니다.",
    hardware: "메모리 대역폭이 병목인 연산이므로, 별도 커널 런치 없이 선행 연산의 레지스터 쓰기(Store) 직전 단계에 ALU 연산을 끼워 넣습니다."
  },
  
  canonical: {
    formula: "Y = X + \\mathbf{b}_{broadcast}",
    shapes: { X: "M x N", b: "1 x N", Y: "M x N" },
    interpretation: {
      M: "입력 샘플 (Batch Size)",
      N: "특징 채널 (Feature Dimension)",
      b: "채널별 보정값 (Bias Term)",
      y_ij: "보정된 활성 신호 (Calibrated Signal)",
    },
  },

  semantics: {
    thesis: "결정 경계(Decision Boundary)를 미세 조정하여 신호의 영점을 맞추는 보정 연산자",
    axes: {
      M: { name: "샘플 (Samples)", role: "신호 처리 단위" },
      N: { name: "특징 채널 (Features)", role: "보정 대상 차원" },
    },

    invariants: [
      {
        id: "INV_DISTRIBUTION_SHIFT",
        name: "분포 평행 이동 (Translation Invariance)",
        metric: "상대적 거리(Relative Distance) 보존",
        threshold: "분산(Variance) 변화량 0",
        allows: ["벡터화된 로드", "스트리밍 융합"],
      },
      {
        id: "INV_ACTIVATION_THRESHOLD",
        name: "활성 임계점 보장 (Threshold Integrity)",
        metric: "부호 반전 비율 (Sign Flip Ratio)",
        threshold: "허용 오차 범위 내 유지",
        allows: ["고정밀도 누산기 사용"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "ReLU 불감 영역 (Dead-zone)",
          rule: "보정 후 값이 음수(Y < 0)로 확정될 경우, 저장 없이 0으로 처리",
          hint: "메모리 쓰기 생략 (Write Elision)",
        },
        {
          name: "LayerNorm 정규화",
          rule: "다음 연산이 정규화일 경우, Bias를 평균 계산 단계로 흡수 가능",
          hint: "연산 융합 (Op Fusion)",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_Epilogue_BiasAdd",
      reason: [
        "메모리 병목(Memory Bound) 연산: 데이터를 다시 읽지 않고 GEMM 직후 처리",
        "캐시 지역성 극대화: 레지스터에 남아있는 값에 즉시 더하기 수행",
        "쓰기 비용 절감: 중간 버퍼(Intermediate Buffer) 생성 방지",
      ],
      applied_rewrites: ["에필로그 융합 (Epilogue Fusion)", "벡터화 처리 (Vectorization)"],
    },
  },

  kernel: {
    strategy: "스트림 처리 및 벡터화 (Vectorized Streaming)",
    details: [
      { technique: "128bit 벡터 로드", semantic_link: "메모리 대역폭 포화 방지" },
      { technique: "레지스터 인젝션", semantic_link: "ALU 유휴 시간 활용 (Compute Hiding)" },
      { technique: "브로드캐스트 최적화", semantic_link: "b 벡터의 L1 캐시 상주 유도" },
    ],
    metrics: {
      memory_reuse: "N/A (Streaming)",
      throughput: "Memory Bandwidth Limit (약 98%)",
      occupancy: "99%"
    },
  },

  costModel: {
    semanticLoss: "\\mathcal{L}_{calib} = \\lambda_{drift} \\| \\Delta_{boundary} \\| + \\epsilon_{quant}",
    weights_hint: {
      default: { drift: 0.8, quant_error: 0.2 }
    },
    metrics: {
      boundary_shift: "0.001%",
      rel_error: "1e-6"
    }
  },

  performance: {
    latency: {
      pytorch: 0.06,
      torch_compile: 0.03,
      ours: 0.01 // 거의 공짜(Free)에 가까운 융합 성능 강조
    }
  }
};