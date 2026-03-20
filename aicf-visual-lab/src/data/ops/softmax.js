// src/data/ops/softmax.js

export const softmaxData = {
  id: "Softmax",
  category: "가설 경쟁 / 확률적 선택 (Hypothesis Competition)",

  descriptions: {
    oneLine:
      "Softmax는 각 row 내부 후보들의 상대적 logit을 정규화된 확률 분포로 변환하는 row-wise competitive normalization operator입니다.",
    essence:
      "Softmax는 각 후보의 logit을 정규화된 확률 분포로 변환하여, 동일 행(row) 안의 후보들이 서로 경쟁하는 선택 구조를 만드는 확률화 연산입니다. 값의 절대 크기보다 상대적 차이가 선택 확률을 결정합니다.",
    strategy:
      "Softmax는 row-wise max reduction, exponentiation, normalization이 결합된 구조이므로, 수치 안정성을 유지하면서 통계 계산과 정규화를 결합하는 lowering이 중요합니다. 특히 후행 연산이 attention-style weighted sum이면 확률 행렬 자체를 별도 저장하지 않는 realization이 가능해집니다.",
    realization:
      "주로 row-wise stabilized normalization family로 연결되며, online max/sum update와 streaming normalization이 중요합니다. fused attention-style path, exp approximation, intermediate materialization 제거의 상세 메커니즘은 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: [
      "m_i = \\max_j x_{i,j}",
      "p_{i,j} = \\frac{e^{x_{i,j} - m_i}}{\\sum_{k=1}^{N} e^{x_{i,k} - m_i}}",
    ].join("\\\\"),
    shapes: {
      x: "M x N",
      p: "M x N",
      m: "M x 1 (Row-wise Stabilizer)",
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
      M: {
        name: "Queries",
        role: "독립적인 경쟁이 수행되는 row/query 축",
        description:
          "각 row는 서로 독립적인 경쟁과 정규화가 수행되는 단위입니다.",
      },
      N: {
        name: "Candidates",
        role: "각 row 안에서 경쟁하는 후보 축",
        description:
          "후보 축은 max, exp-sum, normalization이 수행되는 row-local competition domain입니다.",
      },
    },
  },

  propertyProfile: {
    associative_merge: {
      status: "strong",
      score: 0.94,
      summary:
        "online max / exp-sum state는 partial row chunk 간 병합이 가능하여 streaming merge와 잘 맞습니다.",
      reason:
        "\\text{Row-wise softmax statistics admit associative online / partial merge through stabilized max-sum state updates}",
      allows: [
        "Online Stabilized Update",
        "Partial Row Merge",
        "Streaming Statistic Merge",
      ],
    },

    local_accumulable: {
      status: "conditional",
      score: 0.74,
      summary:
        "row chunk 단위로 local max/sum 상태 누적은 가능하지만 최종 정규화는 row 전체 상태가 확정된 뒤에만 가능합니다.",
      reason:
        "\\text{Local accumulation is valid for row-wise max / exp-sum state, but exact normalization requires finalized row statistics}",
      allows: [
        "Row-Chunk Accumulation",
        "Online Max/Sum Update",
        "Streaming Normalization",
      ],
    },

    tile_composable: {
      status: "conditional",
      score: 0.69,
      summary:
        "candidate 축을 tile로 나눌 수 있지만 tile별 partial softmax state를 정확히 병합해야 합니다.",
      reason:
        "\\text{Candidate-axis tiling is valid only if partial stabilized states are merged without changing row semantics}",
      allows: [
        "Tilewise Row Reduction",
        "Blockwise Candidate Partition",
        "Streaming Tile Merge",
      ],
    },

    order_rewritable: {
      status: "limited",
      score: 0.43,
      summary:
        "candidate 처리 순서는 바꿀 수 있지만 동일 row에 속한 경쟁 집합은 유지되어야 합니다.",
      reason:
        "\\text{Processing order over candidates may vary, but row membership and competitive set semantics must remain fixed}",
      allows: ["Candidate Reordering", "Row-Parallel Scheduling"],
    },

    schedule_invariant: {
      status: "conditional",
      score: 0.66,
      summary:
        "row 내부의 reduction/normalization 스케줄은 유연하지만, 최종 확률은 확정된 stabilized state를 기준으로 계산되어야 합니다.",
      reason:
        "\\text{Execution schedule may vary, but final probabilities must use finalized stabilized row statistics}",
      allows: [
        "Reduction + Normalize Pipeline",
        "Streaming Consumption Schedule",
        "Online Row Schedule",
      ],
    },

    representation_invariant: {
      status: "medium",
      score: 0.71,
      summary:
        "확률 행렬을 반드시 별도 materialize하지 않고 후행 weighted sum에 바로 소비할 수 있습니다.",
      reason:
        "\\text{Softmax output representation may remain implicit if consumed immediately by a downstream weighted-sum operator}",
      allows: [
        "Probability Elision",
        "Softmax-Value Fusion",
        "Streaming Consumer Path",
      ],
    },

    rematerializable: {
      status: "strong",
      score: 0.83,
      summary:
        "softmax probability는 logits에서 재계산 가능해 intermediate materialization 제거와 잘 맞습니다.",
      reason:
        "p_{i,j} \\text{ is rematerializable from logits and stabilized row statistics, enabling intermediate elimination}",
      allows: [
        "Probability Recompute",
        "Intermediate Elimination",
        "FlashAttention-Style Realization",
      ],
    },

    precision_relaxable: {
      status: "conditional",
      score: 0.52,
      summary:
        "exp approximation과 reduced precision은 가능하지만 simplex consistency와 rank sensitivity를 해치면 안 됩니다.",
      reason:
        "\\text{Approximate math is possible if simplex consistency, rank preservation, and distribution shape remain bounded}",
      allows: ["Approximate Exp", "Reduced Precision Accumulation"],
    },

    domain_prunable: {
      status: "conditional",
      score: 0.39,
      summary:
        "exact softmax는 전체 후보를 필요로 하지만, top-k 집중 분포나 sparse regime에서는 근사 pruning이 가능합니다.",
      reason:
        "\\text{Exact softmax requires the full candidate domain, though sparse / top-k approximations may prune low-mass candidates}",
      allows: ["Top-K Approximation", "Sparse Candidate Pruning"],
    },
  },

  opConstraints: [
    {
      id: "TRANSLATION_INVARIANCE",
      name: "이동 불변성 (Translation Invariance)",
      detail:
        "각 row에 동일 상수를 더하거나 빼도 softmax 결과는 변하지 않으며, 이것이 max-subtraction stabilized realization의 수학적 근거입니다.",
      metric:
        "\\mathrm{Softmax}(x_i) = \\mathrm{Softmax}(x_i - c_i \\mathbf{1})",
      consequence: ["Max Subtraction", "Online Stabilized Update"],
    },
    {
      id: "SIMPLEX_CONSTRAINT",
      name: "확률 단체성 (Simplex Constraint)",
      detail:
        "각 row의 출력은 항상 합이 1인 확률 분포를 형성해야 하며 normalization 정확도가 핵심입니다.",
      metric: "\\sum_{j=1}^{N} p_{i,j} = 1",
      consequence: [
        "Row Normalization Fusion",
        "Probability-Constrained Approximation",
      ],
    },
    {
      id: "ORDER_MONOTONICITY",
      name: "순위 단조성 (Order Monotonicity)",
      detail:
        "exact softmax에서는 더 큰 logit이 더 큰 확률을 가져야 하며, 근사화 시 rank inversion이 생기지 않도록 주의해야 합니다.",
      metric:
        "x_{i,j_1} > x_{i,j_2} \\Rightarrow p_{i,j_1} > p_{i,j_2}",
      consequence: ["Top-K Approximation", "Sparse Candidate Pruning"],
    },
  ],

  downstreamConstraints: [
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

  lowering: {
    preferredFamily: "Row-Wise Online Normalization",
    candidates: [
      "Two-Pass_Softmax",
      "Online_Softmax_Fused",
      "Softmax_Value_Fusion",
    ],
    chosen: {
      variant: "Online_Softmax_Fused",
      summary:
        "Softmax는 row-wise max와 exp-sum normalization이 결합된 경쟁 구조를 가지며, translation invariance를 활용한 stabilized online realization과 후행 weighted sum과의 streaming fusion이 자연스럽습니다.",
      reason: [
        "\\text{row-wise 경쟁 구조: } \\max \\text{ 와 } \\sum \\exp(\\cdot) \\text{ 는 동일 row 내부에서 결합된 reduction으로 계산된다}",
        "\\text{이동 불변성: row별 상수 이동은 확률 결과를 바꾸지 않으므로 stabilized realization이 가능하다}",
        "\\text{정규화 결합성: max/sum 계산과 확률 정규화를 streaming 형태로 결합할 수 있다}",
      ],
      applied_rewrites: [
        "Online Stabilized Update",
        "Row-Wise Reduction Fusion",
        "Streaming Normalization",
      ],
    },
  },

  realizationSnapshot: {
    family: "Row-Wise Online Normalization",
    highlights: [
      "Online max/sum update",
      "Row-wise collective reduction",
      "Streaming normalization",
      "Fused weighted-sum consumption path",
    ],
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

  competitionFamily: {
    scope: "row-local",
    candidateAxis: "N",
    stabilizer: "row-max",
    normalizationTarget: "probability-simplex",
    outputMaterialization: "optional",
    fusedConsumer: "weighted-sum",
  },

  cudaCode: `// AICF: Online Softmax (Streaming Realization)
__global__ void online_softmax_kernel(...) {
  // 1. Maintain row-wise running max and running exp-sum
  // 2. Apply numerically stable update while streaming logits
  // 3. Normalize probabilities in the same realization path
  // 4. Optionally consume probabilities immediately in the next op
}`,
};