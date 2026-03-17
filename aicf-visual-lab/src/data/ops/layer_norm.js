// src/data/layer_norm.js

export const layerNormData = {
  id: "LayerNorm",
  category: "분포 재매개변수화 / 표현 안정화 (Distribution Reparameterization)",

  descriptions: {
    essence:
      "LayerNorm은 각 샘플 내부의 feature 축 통계를 기준으로 평균을 제거하고 분산을 정규화하여, 표현의 절대 스케일을 줄이고 상대적 구조를 안정화하는 샘플 단위 정규화 연산입니다.",
    strategy:
      "LayerNorm은 sample-local reduction과 output-local affine transform이 결합된 구조이므로, mean/variance 계산과 normalization, affine 적용을 하나의 realization으로 묶는 lowering이 중요합니다. 핵심은 통계 계산의 정확성을 유지하면서 추가 메모리 왕복을 줄이는 것입니다.",
    hardware:
      "이 연산은 보통 row-wise reduction + pointwise affine family로 연결되며, 실제 one-pass statistics, warp/block reduction, vectorized I/O 같은 구현 세부는 Deep Dive 계층에서 다룹니다.",
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
      M: { name: "Samples", role: "독립적 통계 산출 및 정규화 단위" },
      N: { name: "Features", role: "정규화가 수행되는 내부 feature 축" },
    },

    invariants: [
      {
        id: "INV_SAMPLE_LOCAL_STATISTICS",
        name: "샘플 국소 통계성 (Sample-Local Statistics)",
        metric:
          "\\mu_i, \\sigma_i^2 \\text{ are computed only from features of sample } i",
        threshold: "Per-sample statistic consistency",
        allows: ["Row-Wise Reduction", "One-Pass Statistics"],
      },
      {
        id: "INV_MEAN_CENTERING",
        name: "평균 중심화 (Mean Centering)",
        metric:
          "\\frac{1}{N} \\sum_{j=1}^{N} \\left(x_{i,j} - \\mu_i\\right) = 0",
        threshold: "Centered feature distribution",
        allows: ["Fused Normalization", "Reduction Reordering"],
      },
      {
        id: "INV_AFFINE_RESTORE",
        name: "Affine 복원성 (Affine Restore)",
        metric:
          "y_{i,j} = \\gamma_j \\hat{x}_{i,j} + \\beta_j",
        threshold: "Feature-wise affine consistency",
        allows: ["Affine Fusion", "Vectorized Affine Apply"],
      },
      {
        id: "INV_DENOM_SAFETY",
        name: "분모 안정성 (Denominator Safety)",
        metric: "\\sigma_i^2 + \\epsilon > 0",
        threshold: "Strict Positive",
        allows: ["Epsilon Floor", "Stable rsqrt Approximation"],
      },
    ],

    sensitivity: {
      downstream: [
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
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_LayerNorm_Welford",
      reason: [
        "\\text{샘플 국소 통계 구조: } \\mu_i, \\sigma_i^2 \\text{ 는 각 샘플의 feature 축에서만 계산되므로 row-wise reduction realization이 자연스럽다}",
        "\\text{정규화와 affine 결합: statistics 계산 이후 normalization과 affine apply를 하나의 패스로 유지할 수 있다}",
        "\\text{수치 안정성 요구: variance 계산은 stable online statistics family와 잘 맞는다}",
        "\\text{따라서 } \\texttt{Fused\\_LayerNorm\\_Welford} \\text{ family가 적합하다}",
      ],
      applied_rewrites: [
        "One-Pass Welford Statistics",
        "Row-Wise Reduction Fusion",
        "Vectorized Affine Apply",
      ],
    },
  },

  kernel: {
    strategy: "Row-Wise Reduction & Affine Fusion",
    details: [
      {
        technique: "One-Pass Welford Statistics",
        semantic_link: "샘플별 mean/variance를 안정적으로 계산",
      },
      {
        technique: "Warp / Block Reduction",
        semantic_link: "feature 축 reduction을 sample-local하게 수행",
      },
      {
        technique: "Vectorized Load/Store",
        semantic_link: "연속 feature 구간의 normalization 및 affine 적용 효율화",
      },
      {
        technique: "Fused Affine Apply",
        semantic_link: "정규화 직후 gamma/beta 적용을 결합",
      },
    ],
    metrics: {
      memory_reuse: "Higher than Two-Pass",
      throughput: "Memory Bound / Reduction Dominant",
      occupancy: "High",
    },
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
};