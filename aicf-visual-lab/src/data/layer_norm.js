// src/data/layer_norm.js

export const layerNormData = {
  id: "LayerNorm",
  category: "분포 재매개변수화 / 표현 안정화 (Distribution QA)",

  descriptions: {
    essence: "개별 샘플 내에서 특징 간의 상대적 비율만을 남기고 절대적인 에너지 크기를 제거하여, 학습의 안정을 보장하는 분포 제어 장치입니다.",
    strategy: "데이터 분포의 통계적 의존성을 분석하여, 메모리 재접근 없는 One-Pass 알고리즘(Welford)이나 융합 커널을 선택해 대역폭 병목을 해소합니다.",
    hardware: "Warp Shuffle Intrinsics를 사용하여 스레드 간 통신 비용을 최소화하고, 벡터화된 로드/스토어로 메모리 버스를 100% 활용합니다."
  },

  canonical: {
    formula: "y = \\gamma \\cdot \\frac{x - \\mu}{\\sqrt{\\sigma^2 + \\epsilon}} + \\beta",
    shapes: {
      x: "M x N",
      "\\mu, \\sigma": "M x 1 (Reductions)",
      "\\gamma, \\beta": "1 x N (Parameters)",
      y: "M x N"
    },
    interpretation: {
      x: "입력 특성 (Raw Features)",
      "\\mu, \\sigma": "샘플별 통계량 (Statistics)",
      "\\gamma, \\beta": "학습 가능한 스케일 및 이동 (Affine Restore)",
      "\\epsilon": "수치 안정성 상수 (Epsilon)",
      y: "안정화된 표현 (Stabilized Output)",
    },
  },

  semantics: {
    thesis: "입력 데이터의 절대적 크기(에너지)를 제거하고, 상대적 비율 관계만을 남겨 학습 불안정성을 제어하는 연산자",

    axes: {
      M: { name: "Samples", role: "독립적 통계 산출 단위" },
      N: { name: "Features", role: "정규화 대상 차원" },
    },

    invariants: [
      {
        id: "INV_DISTRIBUTION_STABILITY",
        name: "통계적 분포 안정성 (Statistical Stability)",
        metric: "KL 발산 (KL Divergence)",
        threshold: "분포 오차 < 1e-5",
        allows: ["고속 역제고근(rsqrt) 근사", "Warp-Shuffle 리덕션"],
      },
      {
        id: "INV_AFFINE_INTEGRITY",
        name: "선형 관계 보존 (Affine Integrity)",
        metric: "상관계수 (Correlation Coefficient)",
        threshold: "0.9999 이상",
        allows: ["연산 순서 재배치 (Reordering)"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "Attention 메커니즘",
          rule: "Query/Key 정규화 시 미세한 오차가 Attention Score를 크게 왜곡할 수 있음",
          hint: "정밀도 우선 모드 (High-Precision Accumulation)",
        },
        {
          name: "잔차 연결 (Residual Add)",
          rule: "입력이 Residual Block의 결과일 경우, 메모리 접근 패턴이 유사하므로 융합 유리",
          hint: "Add+LN 융합 (Fused Add-LN)",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_LayerNorm_Welford",
      reason: [
        "메모리 대역폭 제한(Memory Bound): 데이터를 두 번 읽지 않고 평균/분산을 한 번에 계산 (One-Pass)",
        "Welford 알고리즘 적용: 수치적 안정성을 유지하며 통계량 산출",
        "레지스터 셔플링: 공유 메모리(Shared Mem) 대신 레지스터 간 통신으로 지연 시간 최소화",
      ],
      applied_rewrites: ["원패스 통계 산출 (One-Pass Welford)", "벡터화된 로드/스토어"],
    },
  },

  kernel: {
    strategy: "Warp-Level Reduction & Vectorized I/O",
    details: [
      { technique: "벡터화된 메모리 접근", semantic_link: "입출력 병목 해소 (128-bit Access)" },
      { technique: "Welford 온라인 알고리즘", semantic_link: "데이터 재로딩 없이 정확한 표준편차 계산" },
      { technique: "Warp Shuffle", semantic_link: "스레드 간 동기화 비용 최소화" },
    ],
    metrics: {
      memory_reuse: "2.0x (vs Two-Pass)",
      throughput: "Device Limit (95%)",
      occupancy: "88%"
    },
  },

  costModel: {
    semanticLoss: "\\mathcal{L}_{stab} = \\| \\mu - \\hat{\\mu} \\|^2 + \\lambda \\| \\sigma - \\hat{\\sigma} \\|^2",
    weights_hint: {
      default: { mean_error: 10.0, var_error: 10.0 }
    },
    metrics: {
      rel_error: "1e-5",
      stability_score: "High"
    }
  },

  performance: {
    latency: {
      pytorch: 0.45,
      torch_compile: 0.32,
      ours: 0.12 // One-Pass 알고리즘 + 커널 융합 효과
    }
  }
};