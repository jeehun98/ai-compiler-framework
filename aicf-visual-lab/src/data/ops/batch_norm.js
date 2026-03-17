// src/data/batch_norm.js

export const batchNormData = {
  id: "BatchNorm",
  category: "집단 통계 정렬 / 분포 계약 (Collective Distribution Contract)",

  descriptions: {
    essence:
      "BatchNorm은 개별 샘플의 절대값이 아니라 배치 집단의 평균과 분산을 기준으로 활성값을 재정렬하여, 학습 중 표현 분포를 안정화하는 집단 통계 기반 정규화 연산입니다.",
    strategy:
      "BatchNorm은 학습 시 batch statistics, affine transform, running statistics update를 함께 다루는 상태성 연산이며, 추론 시에는 선행 선형 연산과의 수학적 결합을 통해 별도 연산 노드 없이 소거될 수 있습니다.",
    hardware:
      "이 연산은 training에서는 collective-statistics realization으로, inference에서는 folded-erasure realization으로 이어질 수 있으며, 실제 동기화 비용과 memory schedule은 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: [
      "\\mu_B = \\frac{1}{m} \\sum_{i=1}^{m} x_i",
      "\\sigma_B^2 = \\frac{1}{m} \\sum_{i=1}^{m} (x_i - \\mu_B)^2",
      "\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma_B^2 + \\epsilon}}",
      "y_i = \\gamma \\hat{x}_i + \\beta",
      "\\text{RunningMean}_{t+1} = (1-\\alpha) \\cdot \\text{RunningMean}_t + \\alpha \\cdot \\mu_B",
      "\\text{RunningVar}_{t+1} = (1-\\alpha) \\cdot \\text{RunningVar}_t + \\alpha \\cdot \\sigma_B^2",
      "\\text{(Inference)} \\quad y = w_{fold} x + b_{fold}",
    ].join("\\\\"),
    shapes: {
      x: "B x C x H x W",
      "\\mu_B, \\sigma_B^2": "1 x C (Per-Channel Batch Stats)",
      "\\gamma, \\beta": "1 x C (Learnable Affine Params)",
      "RunningMean, RunningVar": "1 x C (State for Inference)",
      y: "B x C x H x W",
    },
    interpretation: {
      x: "현재 배치에서 관측된 활성값",
      "\\mu_B, \\sigma_B^2": "현재 집단이 형성하는 기준 분포",
      "\\gamma, \\beta": "정규화 이후 표현력을 복원하는 affine 파라미터",
      "RunningMean, RunningVar": "추론 시 사용할 장기 통계 상태",
      "Folded": "선행 연산에 흡수되어 별도 노드가 사라진 형태",
    },
  },

  semantics: {
    thesis:
      "BatchNorm은 batch 집단에서 계산된 통계를 기준으로 각 채널의 분포를 정렬하고, affine transform 및 running statistics update를 통해 학습 안정성과 추론 일관성을 동시에 유지하는 collective normalization operator입니다.",

    axes: {
      C: { name: "Channels", role: "독립적 정규화 계약 단위" },
      B: { name: "Batch", role: "현재 통계 기준을 형성하는 집단 축" },
      H: { name: "Spatial Height", role: "채널 통계 집계에 포함되는 공간 축" },
      W: { name: "Spatial Width", role: "채널 통계 집계에 포함되는 공간 축" },
    },

    invariants: [
      {
        id: "INV_CHANNEL_STAT_CONTRACT",
        name: "채널별 통계 계약 (Channel Statistic Contract)",
        metric:
          "\\mu_B, \\sigma_B^2 \\text{ are computed independently for each channel } c",
        threshold: "Per-channel statistic consistency",
        allows: ["Channel-Wise Reduction", "Persistent Reduction"],
      },
      {
        id: "INV_AFFINE_EQUIVALENCE",
        name: "Affine 동치성 (Affine Equivalence)",
        metric:
          "y = \\gamma \\cdot \\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}} + \\beta",
        threshold: "Exact affine restoration",
        allows: ["Conv-BN Folding", "Inference Erasure"],
      },
      {
        id: "INV_RUNNING_STATE_CONSISTENCY",
        name: "Running State 일관성 (Running State Consistency)",
        metric:
          "\\text{RunningStats}_{t+1} \\leftarrow (1-\\alpha)\\text{RunningStats}_t + \\alpha\\text{BatchStats}_t",
        threshold: "EMA-consistent update",
        allows: ["Training-State Update Fusion", "Inference Bridge"],
      },
      {
        id: "INV_DENOM_SAFETY",
        name: "분모 안정성 (Denominator Safety)",
        metric: "\\sigma_B^2 + \\epsilon > 0",
        threshold: "Strict Positive",
        allows: ["Epsilon Floor", "Stable rsqrt Approximation"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "Small Batch Regime",
          rule:
            "B \\text{ 가 매우 작으면 } \\mu_B, \\sigma_B^2 \\text{ 의 추정 오차가 커져 정규화 효과가 불안정해진다}",
          hint: "소배치 환경에서는 GroupNorm/LayerNorm 계열 검토",
        },
        {
          name: "Distributed Sync Requirement",
          rule:
            "\\text{다중 장치 학습에서 전역 batch 통계를 유지하려면 장치 간 statistic synchronization이 필요하다}",
          hint: "SyncBatchNorm 및 통신-연산 overlap 고려",
        },
        {
          name: "Inference Folding Opportunity",
          rule:
            "\\text{선행 Conv/Linear와 affine-normalization 식이 결합 가능하면 BatchNorm 노드를 추론 그래프에서 제거할 수 있다}",
          hint: "Inference graph folding 우선",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Training: Fused_SyncBatchNorm | Inference: Folded_Erasure",
      reason: [
        "\\text{집단 통계 결합(Collective Statistic Coupling): } \\mu_B, \\sigma_B^2, \\gamma, \\beta, \\text{running stats} \\text{ 가 채널 기준으로 강하게 연결된다}",
        "\\text{학습 시 } \\text{statistics reduction, normalization, affine transform, running-state update를 결합된 realization으로 유지할 수 있다}",
        "\\text{분산 학습에서는 global batch semantics 유지를 위해 synchronized statistics가 필요하다}",
        "\\text{추론 시에는 affine-normalization 식이 선행 Conv/Linear와 합성 가능하므로 } \\texttt{Folded\\_Erasure} \\text{ family가 성립한다}",
      ],
      applied_rewrites: [
        "Persistent CTA Reduction",
        "Sync Statistics Fusion",
        "Conv-BN Folding (Inference)",
      ],
    },
  },

  kernel: {
    strategy: "Persistent CTA Reduction & Statistic Sync",
    details: [
      {
        technique: "Persistent Thread Block",
        semantic_link: "채널별 통계 계산과 정규화를 재로딩 최소화 형태로 결합",
      },
      {
        technique: "Warp / Block Reduction",
        semantic_link: "채널 통계를 collective reduction 형태로 계산",
      },
      {
        technique: "Cross-Device Statistic Sync",
        semantic_link: "분산 학습 시 global batch semantics 유지",
      },
      {
        technique: "Inference Folding",
        semantic_link: "추론 그래프에서 BatchNorm 노드 소거",
      },
    ],
    metrics: {
      memory_reuse: "High (Persistent)",
      throughput: "Sync Bound (Distributed) / Memory Bound (Local)",
      occupancy: "85%",
    },
  },

  costModel: {
    semanticLoss:
      "\\mathcal{C}_{bn} = w_{stat} \\cdot \\Delta_{stat} + w_{sync} \\cdot \\Delta_{sync} + w_{infer} \\cdot \\Delta_{fold}",
    weights_hint: {
      default: {
        stat: 45.0,
        sync: 30.0,
        infer: 15.0,
        stability: 10.0,
      },
    },
    metrics: {
      statistic_consistency: "High",
      sync_sensitivity: "Moderate-High",
      inference_erasure_affinity: "Strong",
    },
  },

  performance: {
    latency: {
      pytorch: 0.5,
      torch_compile: 0.35,
      ours: 0.15,
    },
    notes: {
      inference: "Folded inference path can erase the standalone BatchNorm kernel.",
    },
  },

  cudaCode: `// AICF: Fused SyncBatchNorm (Training)
__global__ void batch_norm_train(...) {
  // 1. Per-channel local reduction for sum / square-sum
  // 2. Optional cross-device statistic synchronization
  // 3. Compute mean / variance
  // 4. Normalize + affine transform
  // 5. Update running statistics in the same realization path
}

// Inference path:
// BatchNorm is folded into the preceding Conv/Linear weights and bias,
// so no standalone BatchNorm kernel is launched.`,
};