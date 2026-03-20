// src/data/ops/layer_norm.js

export const layerNormData = {
  id: "LayerNorm",
  category: "분포 재매개변수화 / 표현 안정화 (Distribution Reparameterization)",

  descriptions: {
    oneLine:
      "LayerNorm은 각 샘플 내부 feature 축 통계를 기준으로 표현을 재중심화·재스케일링하고 affine transform을 적용하는 sample-local normalization operator입니다.",
    essence:
      "LayerNorm은 각 샘플 내부의 feature 축 통계를 기준으로 평균을 제거하고 분산을 정규화하여, 표현의 절대 스케일을 줄이고 상대적 구조를 안정화하는 샘플 단위 정규화 연산입니다.",
    strategy:
      "LayerNorm은 sample-local reduction과 output-local affine transform이 결합된 구조이므로, mean/variance 계산과 normalization, affine 적용을 하나의 realization으로 묶는 lowering이 중요합니다. 핵심은 통계 계산의 정확성을 유지하면서 추가 메모리 왕복을 줄이는 것입니다.",
    realization:
      "주로 row-wise reduction + fused affine family로 연결되며, one-pass statistics와 numerically stable variance evaluation이 중요합니다. warp/block reduction, vectorized I/O, stable online statistics의 상세 구현은 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: [
      "\\mu_i = \\frac{1}{N} \\sum_{j=1}^{N} x_{i,j}",
      "\\sigma_i^2 = \\frac{1}{N} \\sum_{j=1}^{N} (x_{i,j} - \\mu_i)^2",
      "y_{i,j} = \\gamma_j \\cdot \\frac{x_{i,j} - \\mu_i}{\\sqrt{\\sigma_i^2 + \\epsilon}} + \\beta_j",
    ].join("\\\\"),
    shapes: {
      x: "M x N",
      "\\mu, \\sigma^2": "M x 1 (Per-Sample Statistics)",
      "\\gamma, \\beta": "1 x N (Affine Parameters)",
      y: "M x N",
    },
    interpretation: {
      M: "독립적으로 정규화되는 샘플/토큰 축",
      N: "정규화가 수행되는 feature 축",
      x: "입력 표현",
      "\\mu, \\sigma^2": "샘플 내부 feature 분포 통계",
      "\\gamma, \\beta": "정규화 후 표현 복원을 위한 affine 파라미터",
      y: "안정화된 출력 표현",
    },
  },

  semantics: {
    thesis:
      "LayerNorm은 각 샘플 내부 feature 축의 통계를 사용해 표현을 재중심화하고 재스케일링하는 sample-local normalization operator이며, sequence 길이나 batch 구성과 독립적으로 안정적인 표현 분포를 유지하는 데 사용됩니다.",
    axes: {
      M: {
        name: "Samples",
        role: "독립적 통계 산출 및 정규화 단위",
        description:
          "각 샘플 또는 토큰은 자신의 feature 축만으로 평균과 분산을 계산합니다.",
      },
      N: {
        name: "Features",
        role: "정규화가 수행되는 내부 feature 축",
        description:
          "feature 축은 샘플 내부 통계 계산과 affine 적용이 이루어지는 주된 reduction domain입니다.",
      },
    },
  },

  propertyProfile: {
    associative_merge: {
      status: "strong",
      score: 0.89,
      summary:
        "row 내부 feature 축 reduction은 partial sum / partial variance merge 형태로 누적 가능합니다.",
      reason:
        "\\text{Per-sample statistics over feature axis } N \\text{ can be formed through associative partial reduction / merge}",
      allows: [
        "Partial Statistic Merge",
        "Tree Reduction",
        "Welford Merge",
      ],
    },

    local_accumulable: {
      status: "strong",
      score: 0.93,
      summary:
        "각 sample row 안에서 local partial statistics를 누적하고 최종 normalization을 이어서 수행할 수 있습니다.",
      reason:
        "\\mu_i, \\sigma_i^2 \\text{ are computed entirely from the local feature domain of sample } i",
      allows: [
        "Row-Wise Reduction",
        "One-Pass Statistics",
        "Fused Normalize + Affine",
      ],
    },

    tile_composable: {
      status: "conditional",
      score: 0.64,
      summary:
        "feature 축을 타일로 나눌 수 있지만 tile 간 partial statistics merge가 정확히 보존되어야 합니다.",
      reason:
        "\\text{Feature-axis tiling is valid only if partial row statistics are merged without changing per-sample semantics}",
      allows: [
        "Feature Tile Reduction",
        "Warp/Block Partitioned Row Reduction",
      ],
    },

    order_rewritable: {
      status: "limited",
      score: 0.47,
      summary:
        "feature reduction 순서는 바꿀 수 있지만 sample-local membership은 유지되어야 합니다.",
      reason:
        "\\text{Reduction order over } N \\text{ may vary, but the feature set belonging to each sample } i \\text{ must remain fixed}",
      allows: ["Reduction Reordering", "Row-Parallel Scheduling"],
    },

    schedule_invariant: {
      status: "conditional",
      score: 0.61,
      summary:
        "부분 reduction과 affine apply의 내부 스케줄은 바뀔 수 있지만 통계 확정 후 normalize 순서는 유지되어야 합니다.",
      reason:
        "\\text{Execution schedule may vary, but final normalization must use finalized per-sample statistics}",
      allows: [
        "Warp/Block Reduction Schedule",
        "Reduction + Normalize Pipeline",
      ],
    },

    representation_invariant: {
      status: "strong",
      score: 0.81,
      summary:
        "vector width, row layout, packed I/O 변화는 가능하지만 sample-local reduction 의미는 유지되어야 합니다.",
      reason:
        "\\text{Layout / vectorization changes are allowed if row boundaries and per-sample feature semantics are preserved}",
      allows: [
        "Vectorized Load/Store",
        "Packed Affine Apply",
        "Contiguous Row Layout Preference",
      ],
    },

    rematerializable: {
      status: "medium",
      score: 0.58,
      summary:
        "mean/variance는 입력 row에서 재계산 가능해 intermediate 저장을 줄일 수 있습니다.",
      reason:
        "\\mu_i, \\sigma_i^2 \\text{ are recomputable from } x_i \\text{ without external running state}",
      allows: ["Statistic Recompute", "Saved-Intermediate Reduction"],
    },

    precision_relaxable: {
      status: "conditional",
      score: 0.56,
      summary:
        "fast rsqrt 및 mixed precision은 가능하지만 variance 정확도와 denominator stability를 해치면 안 됩니다.",
      reason:
        "\\text{Approximate math is possible if mean/variance consistency and denominator stability remain bounded}",
      allows: ["Stable rsqrt Approximation", "Mixed Precision Reduction"],
    },

    domain_prunable: {
      status: "weak",
      score: 0.16,
      summary:
        "일반 LayerNorm은 전체 feature domain 참여가 필요하므로 pruning 여지가 작습니다.",
      reason:
        "\\text{LayerNorm generally needs the full feature domain of each sample to compute correct statistics}",
      allows: [],
    },
  },

  opConstraints: [
    {
      id: "SAMPLE_LOCAL_STATISTICS",
      name: "샘플 국소 통계성 (Sample-Local Statistics)",
      detail:
        "각 샘플의 평균과 분산은 해당 샘플 내부 feature들만으로 계산되어야 하며 다른 샘플과 섞이면 안 됩니다.",
      metric:
        "\\mu_i, \\sigma_i^2 \\text{ are computed only from features of sample } i",
      consequence: ["Row-Wise Reduction", "One-Pass Statistics"],
    },
    {
      id: "MEAN_CENTERING",
      name: "평균 중심화 (Mean Centering)",
      detail:
        "정규화 과정은 각 샘플 내부 feature 분포를 평균 0 기준으로 재중심화해야 합니다.",
      metric:
        "\\frac{1}{N} \\sum_{j=1}^{N} \\left(x_{i,j} - \\mu_i\\right) = 0",
      consequence: ["Fused Normalization", "Reduction Reordering"],
    },
    {
      id: "AFFINE_RESTORE",
      name: "Affine 복원성 (Affine Restore)",
      detail:
        "정규화된 표현은 feature-wise affine transform을 통해 표현력을 복원할 수 있어 fused affine apply가 자연스럽습니다.",
      metric: "y_{i,j} = \\gamma_j \\hat{x}_{i,j} + \\beta_j",
      consequence: ["Affine Fusion", "Vectorized Affine Apply"],
    },
    {
      id: "DENOM_SAFETY",
      name: "분모 안정성 (Denominator Safety)",
      detail:
        "정규화 분모는 epsilon을 포함해 strict positive를 유지해야 하며 variance 계산 오차에 민감합니다.",
      metric: "\\sigma_i^2 + \\epsilon > 0",
      consequence: ["Epsilon Floor", "Stable rsqrt Approximation"],
    },
  ],

  downstreamConstraints: [
    {
      name: "Attention / QKV Projection",
      rule:
        "\\text{LayerNorm 출력이 Q/K/V projection으로 직접 연결되면 작은 통계 오차도 후행 score 분포에 영향을 줄 수 있다}",
      hint: "통계 정확도 우선 및 numerically stable realization",
    },
    {
      name: "Residual Add + LayerNorm Chain",
      rule:
        "\\text{입력이 residual add 결과라면 add와 normalization이 연속된 pointwise-reduction 구조를 이루므로 fused Add+LN lowering이 유리하다}",
      hint: "Residual-aware fusion 검토",
    },
    {
      name: "Large Feature Dimension",
      rule:
        "N \\text{ 이 커질수록 row-wise reduction 비용과 메모리 접근 패턴이 성능에 더 큰 영향을 준다}",
      hint: "One-pass statistics 및 vectorized reduction 우선",
    },
  ],

  lowering: {
    preferredFamily: "Row-Wise Reduction & Affine Fusion",
    candidates: [
      "Two-Pass_LayerNorm",
      "Fused_LayerNorm_Welford",
      "Add_LayerNorm_Fusion",
    ],
    chosen: {
      variant: "Fused_LayerNorm_Welford",
      summary:
        "LayerNorm은 샘플별 feature 축에서 통계를 계산하는 row-wise reduction 구조와 output-local affine apply가 결합된 형태이므로, stable one-pass statistics 기반 fused realization이 자연스럽습니다.",
      reason: [
        "\\text{샘플 국소 통계 구조: } \\mu_i, \\sigma_i^2 \\text{ 는 각 샘플의 feature 축에서만 계산되므로 row-wise reduction realization이 자연스럽다}",
        "\\text{정규화와 affine 결합: statistics 계산 이후 normalization과 affine apply를 하나의 패스로 유지할 수 있다}",
        "\\text{수치 안정성 요구: variance 계산은 stable online statistics family와 잘 맞는다}",
      ],
      applied_rewrites: [
        "One-Pass Welford Statistics",
        "Row-Wise Reduction Fusion",
        "Vectorized Affine Apply",
      ],
    },
  },

  realizationSnapshot: {
    family: "Row-Wise Reduction & Affine Fusion",
    highlights: [
      "One-pass Welford statistics",
      "Sample-local row reduction",
      "Vectorized normalization / affine apply",
      "Lower memory traffic than two-pass realization",
    ],
  },

  costModel: {
    semanticLoss:
      "\\mathcal{C}_{ln} = w_{mean} \\cdot \\Delta_{mean} + w_{var} \\cdot \\Delta_{var} + w_{aff} \\cdot \\Delta_{affine}",
    weights_hint: {
      default: {
        mean: 35.0,
        variance: 40.0,
        affine: 15.0,
        numeric: 10.0,
      },
    },
    metrics: {
      mean_consistency: "High",
      variance_consistency: "High",
      affine_restore_affinity: "Strong",
    },
  },

  performance: {
    latency: {
      pytorch: 0.45,
      torch_compile: 0.32,
      ours: 0.12,
    },
  },

  linearProjectionFamily: {
    contraction: "A[M,K] x B[K,N] -> C[M,N]",
    reductionAxis: "K",
    outputTiling: true,
    epilogueFusion: true,
    tensorCoreFriendly: true,
    outputMaterialization: "preferred-after-epilogue",
  },
};