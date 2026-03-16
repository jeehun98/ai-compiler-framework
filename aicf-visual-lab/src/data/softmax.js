// src/data/softmax.js

export const softmaxData = {
  id: "Softmax",
  category: "가설 경쟁 / 확률적 선택 (Hypothesis Competition)",

  descriptions: {
    essence:
      "Softmax는 각 후보의 logit을 정규화된 확률 분포로 변환하여, 동일 행(row) 안의 후보들이 서로 경쟁하는 선택 구조를 만드는 확률화 연산입니다. 값의 절대 크기보다 상대적 차이가 선택 확률을 결정합니다.",
    strategy:
      "Softmax는 row-wise max reduction, exponentiation, normalization이 결합된 구조이므로, 수치 안정성을 유지하면서 통계 계산과 정규화를 결합하는 lowering이 중요합니다. 특히 후행 연산이 attention-style weighted sum이면 확률 행렬 자체를 별도 저장하지 않는 realization이 가능해집니다.",
    hardware:
      "이 연산은 보통 row-wise reduction + normalization family로 연결되며, 실제 online update, exp approximation, fused attention-style realization 같은 구현 세부는 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: [
      "m_i = \\max_j x_{i,j}",
      "p_{i,j} = \\frac{e^{x_{i,j} - m_i}}{\\sum_{k=1}^{N} e^{x_{i,k} - m_i}}",
    ].join("\\\\"),
    shapes: {
      x: "M x N",
      p: "M x N",
      "m": "M x 1 (Row-wise Stabilizer)",
    },
    interpretation: {
      M: "독립적으로 경쟁이 일어나는 row/query 축",
      N: "경쟁하는 후보/candidate 축",
      x: "정규화 전 경쟁 점수 (logits)",
      p: "정규화된 선택 확률",
      "m_i": "수치 안정성을 위한 row별 기준점",
    },
  },

  semantics: {
    thesis:
      "Softmax는 각 row 내부 후보들의 상대적 에너지를 probability simplex 위의 분포로 변환하는 row-wise competitive normalization operator이며, 순위 구조를 보존하면서 선택 집중도와 엔트로피를 재조정합니다.",

    axes: {
      M: { name: "Queries", role: "독립적인 경쟁이 수행되는 row/query 축" },
      N: { name: "Candidates", role: "각 row 안에서 경쟁하는 후보 축" },
    },

    invariants: [
      {
        id: "INV_TRANSLATION_INVARIANCE",
        name: "이동 불변성 (Translation Invariance)",
        metric:
          "\\mathrm{Softmax}(x_i) = \\mathrm{Softmax}(x_i - c_i \\mathbf{1})",
        threshold: "Exact row-wise equivalence",
        allows: ["Max Subtraction", "Online Stabilized Update"],
      },
      {
        id: "INV_SIMPLEX_CONSTRAINT",
        name: "확률 단체성 (Simplex Constraint)",
        metric: "\\sum_{j=1}^{N} p_{i,j} = 1",
        threshold: "Row-wise normalized distribution",
        allows: ["Row Normalization Fusion", "Probability-Constrained Approximation"],
      },
      {
        id: "INV_ORDER_MONOTONICITY",
        name: "순위 단조성 (Order Monotonicity)",
        metric:
          "x_{i,j_1} > x_{i,j_2} \\Rightarrow p_{i,j_1} > p_{i,j_2}",
        threshold: "No rank inversion under exact softmax",
        allows: ["Top-K Approximation", "Sparse Candidate Pruning"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "Attention Weighted Sum",
          rule:
            "\\text{Softmax 출력이 즉시 value-weighted sum에 사용되면 } p \\text{ 전체를 별도 메모리에 기록하지 않고 streaming realization이 가능하다}",
          hint: "Softmax-V fusion 또는 FlashAttention-style lowering 검토",
        },
        {
          name: "Top-K / Sampling Regime",
          rule:
            "\\text{큰 row에서 일부 상위 후보가 분포 질량 대부분을 차지하면 하위 후보에 대한 근사/생략 전략이 성립할 수 있다}",
          hint: "Sparse / Top-K aware softmax 검토",
        },
        {
          name: "Numeric Range Sensitivity",
          rule:
            "\\text{row 내 logit range가 매우 크면 exponentiation 이전 stabilizing transform의 정확성이 중요하다}",
          hint: "Max-subtracted numerically stable realization 우선",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Online_Softmax_Fused",
      reason: [
        "\\text{row-wise 경쟁 구조: } \\max \\text{ 와 } \\sum \\exp(\\cdot) \\text{ 는 동일 row 내부에서 결합된 reduction으로 계산된다}",
        "\\text{이동 불변성: row별 상수 이동은 확률 결과를 바꾸지 않으므로 stabilized realization이 가능하다}",
        "\\text{정규화 결합성: max/sum 계산과 확률 정규화를 streaming 형태로 결합할 수 있다}",
        "\\text{후행 weighted sum과 연결될 경우 } \\texttt{Online\\_Softmax\\_Fused} \\text{ family가 특히 유리하다}",
      ],
      applied_rewrites: [
        "Online Stabilized Update",
        "Row-Wise Reduction Fusion",
        "Streaming Normalization",
      ],
    },
  },

  kernel: {
    strategy: "Row-Wise Online Normalization",
    details: [
      {
        technique: "Online Max/Sum Update",
        semantic_link: "row 통계를 스트리밍 방식으로 안정적으로 갱신",
      },
      {
        technique: "Register / Shared Reduction",
        semantic_link: "row 내부 후보들에 대한 collective reduction 수행",
      },
      {
        technique: "Fast Exp Approximation",
        semantic_link: "순위 및 분포 품질 허용 범위 내에서 exp 비용 완화",
      },
      {
        technique: "Fused Weighted Sum Path",
        semantic_link: "softmax 결과를 즉시 후행 연산에 소비하여 중간 materialization 제거",
      },
    ],
    metrics: {
      memory_reuse: "High in Streaming/Fused Path",
      throughput: "Reduction + Exp Dominant",
      occupancy: "High",
    },
  },

  costModel: {
    semanticLoss:
      "\\mathcal{C}_{softmax} = w_{norm} \\cdot \\Delta_{simplex} + w_{rank} \\cdot \\Delta_{order} + w_{dist} \\cdot \\Delta_{distribution}",
    weights_hint: {
      default: {
        norm: 35.0,
        rank: 30.0,
        distribution: 35.0,
      },
    },
    metrics: {
      simplex_consistency: "High",
      order_preservation: "High",
      fusion_affinity: "Strong",
    },
  },

  performance: {
    latency: {
      pytorch: 0.15,
      torch_compile: 0.1,
      ours: 0.04,
    },
  },

  cudaCode: `// AICF: Online Softmax (Streaming Realization)
__global__ void online_softmax_kernel(...) {
  // 1. Maintain row-wise running max and running exp-sum
  // 2. Apply numerically stable update while streaming logits
  // 3. Normalize probabilities in the same realization path
  // 4. Optionally consume probabilities immediately in the next op
}`,
};