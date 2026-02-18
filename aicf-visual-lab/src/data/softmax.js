// src/data/softmax.js

export const softmaxData = {
  id: "Softmax",
  category: "가설 경쟁 / 확률적 선택 (Hypothesis Competition)",


  descriptions: {
    essence: "무질서한 로짓(Logit) 값을 확률 공간으로 압축하며, 강한 가설은 살리고 약한 가설은 도태시키는 가설 경쟁(Competition) 시스템입니다.",
    strategy: "Online 알고리즘을 통해 통계 산출(Max/Sum)과 정규화를 단일 패스로 처리하고, 유의미하지 않은 하위 확률을 생략하는 희소성(Sparsity) 최적화를 탐색합니다.",
    hardware: "지수 함수(Exp) 연산 비용을 줄이기 위해 고속 근사(Fast Math) 유닛을 활용하거나, 레지스터 내에서 통계를 갱신하는 Online Update를 적용합니다."
  },

  canonical: {
    formula: "p_i = \\frac{e^{x_i - x_{max}}}{\\sum_j e^{x_j - x_{max}}}",
    shapes: {
      x: "M x N",
      p: "M x N",
      "x_{max}": "M x 1 (Stabilizer)"
    },
    interpretation: {
      x: "경쟁 에너지 (Logits / Raw Scores)",
      p: "생존 확률 (Selection Probability)",
      "x_{max}": "최대 우도 기준점 (Numerical Anchor)",
      sum: "분할 함수 (Partition Function / Normalizer)"
    },
  },

  semantics: {
    thesis: "무질서한 에너지(Logits)를 정규화된 확률로 변환하며, 강한 신호는 증폭하고 약한 신호는 침묵시키는(Silence) 엔트로피 압축기",

    axes: {
      M: { name: "Queries", role: "독립된 경쟁의 장 (Batch)" },
      N: { name: "Candidates", role: "경쟁하는 가설들 (Logit Space)" },
    },

    invariants: [
      {
        id: "INV_TRANSLATION",
        name: "이동 불변성 (Translation Invariance)",
        metric: "Softmax(x) = Softmax(x - c)",
        threshold: "오차 0 (수학적 항등)",
        allows: ["수치 안정성을 위한 Max-Subtraction 기법"],
      },
      {
        id: "INV_PROBABILITY_SUM",
        name: "확률 총합 보존 (Simplex Constraint)",
        metric: "\\sum p_i = 1.0",
        threshold: "|1 - \\sum| < 1e-6",
        allows: ["정규화 상수(Z) 재계산"],
      },
      {
        id: "INV_ORDER_PRESERVE",
        name: "단조 증가성 (Monotonicity)",
        metric: "Rank(x) == Rank(p)",
        threshold: "순위 역전 없음",
        allows: ["Top-K 근사", "희소(Sparse) 연산"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "FlashAttention 패턴",
          rule: "Softmax 결과가 즉시 V와 곱해진다면(Attention), 전체 행렬을 VRAM에 쓸 필요 없음",
          hint: "커널 융합 (Op Fusion) & 타일링(Tiling)",
        },
        {
          name: "Top-K Sampling",
          rule: "상위 K개 이외의 값은 의미적으로 0에 수렴하므로 연산 생략 가능",
          hint: "희소 Softmax (Sparse Softmax)",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Online_Softmax_Fused",
      reason: [
        "메모리 대역폭 절약: N이 클 때 중간 결과(Exponentials)를 메모리에 쓰지 않음",
        "수치 안정성(Numerical Stability): Online 알고리즘으로 Overflow 방지 (Max-Subtraction 자동 적용)",
        "Pass 융합: 통계량(Max, Sum) 계산과 정규화를 하나의 커널 패스로 통합",
      ],
      applied_rewrites: ["Online Welford-style Update", "Loop Fusion", "Register Tiling"],
    },
  },

  kernel: {
    strategy: "Online Softmax (Safe-Mode)",
    details: [
      { technique: "Online Update", semantic_link: "데이터를 스트리밍하며 Max와 Sum을 동적 갱신" },
      { technique: "Register Packing", semantic_link: "FP16/BF16 벡터 연산 가속" },
      { technique: "Exp 근사 (Fast Math)", semantic_link: "의미론적 순위 보존 내에서의 고속 연산 허용" },
    ],
    metrics: {
      memory_reuse: "Maximum (L1 Cache)",
      throughput: "Compute Bound (Exp 연산 비중 높음)",
      occupancy: "94%"
    },
  },

  costModel: {
    semanticLoss: "\\mathcal{L}_{sel} = D_{KL}(P_{ideal} || P_{approx}) + \\lambda \\cdot \\text{Sparsity}",
    weights_hint: {
      default: { kl_div: 50.0, sparsity_bonus: 2.0 }
    },
    metrics: {
      entropy_preservation: "99.9%",
      active_candidates: "~4% (Sparse Regime)"
    }
  },

  performance: {
    latency: {
      pytorch: 0.15,
      torch_compile: 0.10,
      ours: 0.04 // Online Softmax + Fusion 효과
    }
  },

  cudaCode: `// AICF: Online Softmax (Single-Pass)
__global__ void online_softmax_kernel(...) {
  // Update max and sum iteratively in registers
  // Avoid writing full M x N intermediate matrix to HBM
  // Combine with next op (e.g., Dropout/MatMul) if possible
}`
};