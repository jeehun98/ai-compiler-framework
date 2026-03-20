// src/data/ops/batch_norm.js

export const batchNormData = {
  id: "BatchNorm",
  category: "집단 통계 정렬 / 분포 계약 (Collective Distribution Contract)",

  descriptions: {
    oneLine:
      "BatchNorm은 batch 집단 통계를 기준으로 채널별 표현 분포를 정렬하고, 학습 시 running statistics를 갱신하며 추론 시 선행 선형 연산에 fold될 수 있는 collective normalization operator입니다.",
    essence:
      "BatchNorm은 개별 샘플의 절대값이 아니라 배치 집단의 평균과 분산을 기준으로 활성값을 재정렬하여, 학습 중 표현 분포를 안정화하는 집단 통계 기반 정규화 연산입니다.",
    strategy:
      "BatchNorm은 학습 시 batch statistics, affine transform, running statistics update를 함께 다루는 상태성 연산이며, 추론 시에는 선행 선형 연산과의 수학적 결합을 통해 별도 연산 노드 없이 소거될 수 있습니다.",
    realization:
      "학습 시에는 collective-statistics normalization family로, 추론 시에는 folded-erasure family로 이어질 수 있습니다. 동기화 비용과 memory schedule, folding mechanism은 Deep Dive 계층에서 다룹니다.",
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
      Folded: "선행 연산에 흡수되어 별도 노드가 사라진 형태",
    },
  },

  semantics: {
    thesis:
      "BatchNorm은 batch 집단에서 계산된 통계를 기준으로 각 채널의 분포를 정렬하고, affine transform 및 running statistics update를 통해 학습 안정성과 추론 일관성을 동시에 유지하는 collective normalization operator입니다.",
    axes: {
      C: {
        name: "Channels",
        role: "독립적 정규화 계약 단위",
        description:
          "각 채널은 자신의 평균, 분산, affine 파라미터, running state를 기준으로 정규화됩니다.",
      },
      B: {
        name: "Batch",
        role: "현재 통계 기준을 형성하는 집단 축",
        description:
          "배치 축은 현재 step에서 평균과 분산을 형성하는 collective population입니다.",
      },
      H: {
        name: "Spatial Height",
        role: "채널 통계 집계에 포함되는 공간 축",
        description:
          "공간 축은 채널별 통계 계산에 참여하는 reduction domain의 일부입니다.",
      },
      W: {
        name: "Spatial Width",
        role: "채널 통계 집계에 포함되는 공간 축",
        description:
          "공간 축은 채널별 통계 계산에 참여하는 reduction domain의 일부입니다.",
      },
      t: {
        name: "Time",
        role: "running statistics update 기준",
        description:
          "학습 step에 따라 running mean/variance가 EMA 형태로 갱신됩니다.",
      },
    },
  },

  propertyProfile: {
    associative_merge: {
      status: "strong",
      score: 0.91,
      summary:
        "batch statistics는 sum / square-sum reduction으로 누적 가능하여 부분 통계 병합이 자연스럽습니다.",
      reason:
        "\\text{Per-channel batch statistics can be formed via associative partial reductions over } (B,H,W)",
      allows: [
        "Partial Statistic Merge",
        "Tree Reduction",
        "Persistent CTA Reduction",
      ],
    },

    local_accumulable: {
      status: "conditional",
      score: 0.63,
      summary:
        "채널별 local partial sum/square-sum 누적은 가능하지만 최종 normalization은 collective statistics 완성 이후에만 가능",
      reason:
        "\\text{Local accumulation is valid for partial stats, but normalized output requires globally consistent } \\mu_B, \\sigma_B^2",
      allows: [
        "Local Partial Reduction",
        "Shared-Memory Statistic Accumulation",
      ],
    },

    tile_composable: {
      status: "medium",
      score: 0.66,
      summary:
        "입력 도메인을 tile로 나눌 수 있지만 tile 간 partial statistics merge가 반드시 뒤따라야 합니다.",
      reason:
        "\\text{Input tiles over } (B,H,W) \\text{ are valid only if their partial statistics are merged consistently per channel}",
      allows: ["Tilewise Reduction", "Blockwise Statistic Merge"],
    },

    order_rewritable: {
      status: "limited",
      score: 0.44,
      summary:
        "통계 축 내부 순서는 바뀔 수 있지만 reduction membership 자체가 바뀌면 의미가 깨집니다.",
      reason:
        "\\text{Reduction order may vary, but the membership of the normalization domain over } (B,H,W) \\text{ must remain fixed}",
      allows: ["Reduction Reordering", "Channel-Parallel Scheduling"],
    },

    schedule_invariant: {
      status: "conditional",
      score: 0.62,
      summary:
        "부분 reduction 스케줄은 유연하지만 학습 시 running-state update와 통계 확정 순서는 보존되어야 합니다.",
      reason:
        "\\text{Statistic accumulation schedule may vary, but running-state update must follow finalized batch statistics}",
      allows: [
        "Persistent CTA Scheduling",
        "Reduction / Normalize Pipeline",
        "Sync-Aware Schedule",
      ],
    },

    representation_invariant: {
      status: "strong",
      score: 0.84,
      summary:
        "추론 시 BN은 선행 Conv/Linear에 흡수될 수 있어 표현 형태 변경에 대한 동치성이 강합니다.",
      reason:
        "\\text{At inference, BatchNorm can be algebraically absorbed into preceding affine operators without changing semantics}",
      allows: [
        "Conv-BN Folding",
        "Linear-BN Folding",
        "Inference Erasure",
      ],
    },

    rematerializable: {
      status: "limited",
      score: 0.33,
      summary:
        "학습 시 batch statistics를 다시 계산할 수는 있지만 collective reduction 비용이 존재합니다.",
      reason:
        "\\mu_B, \\sigma_B^2 \\text{ are recomputable from } x, \\text{ but rematerialization is not cheap due to collective reduction cost}",
      allows: ["Statistic Recompute"],
    },

    precision_relaxable: {
      status: "conditional",
      score: 0.57,
      summary:
        "rsqrt 등 근사는 가능하지만 작은 분산, 작은 batch, sync 오차에 민감합니다.",
      reason:
        "\\text{Approximate math is possible if channel statistics and denominator stability remain numerically reliable}",
      allows: ["Stable rsqrt Approximation", "Mixed Precision Reduction"],
    },

    domain_prunable: {
      status: "weak",
      score: 0.14,
      summary:
        "일반적인 BatchNorm은 전체 normalization domain을 필요로 하므로 pruning 여지가 작습니다.",
      reason:
        "\\text{BatchNorm generally requires full participation of the normalization domain for correct batch statistics}",
      allows: [],
    },
  },

  opConstraints: [
    {
      id: "CHANNEL_STAT_CONTRACT",
      name: "채널별 통계 계약 (Channel Statistic Contract)",
      detail:
        "평균과 분산은 각 채널별로 독립 계산되어야 하며 채널 간 통계 혼합은 허용되지 않습니다.",
      metric:
        "\\mu_B[c],\\ \\sigma_B^2[c] \\text{ are computed independently for each channel } c",
      consequence: ["Channel-Wise Reduction", "Persistent Reduction"],
    },
    {
      id: "AFFINE_EQUIVALENCE",
      name: "Affine 동치성 (Affine Equivalence)",
      detail:
        "정규화 후 affine transform은 추론 시 선행 Conv/Linear와 결합될 수 있어 folding의 수학적 근거가 됩니다.",
      metric:
        "y = \\gamma \\cdot \\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}} + \\beta",
      consequence: ["Conv-BN Folding", "Inference Erasure"],
    },
    {
      id: "RUNNING_STATE_CONSISTENCY",
      name: "Running State 일관성 (Running State Consistency)",
      detail:
        "running mean/variance는 batch statistics가 확정된 뒤 EMA 규칙으로 일관되게 갱신되어야 합니다.",
      metric:
        "\\text{RunningStats}_{t+1} \\leftarrow (1-\\alpha)\\text{RunningStats}_t + \\alpha\\text{BatchStats}_t",
      consequence: ["Training-State Update Fusion", "Inference Bridge"],
    },
    {
      id: "DENOM_SAFETY",
      name: "분모 안정성 (Denominator Safety)",
      detail:
        "정규화 분모는 epsilon floor를 포함해 strict positive를 유지해야 합니다.",
      metric: "\\sigma_B^2 + \\epsilon > 0",
      consequence: ["Epsilon Floor", "Stable rsqrt Approximation"],
    },
  ],

  downstreamConstraints: [
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

  lowering: {
    preferredFamily:
      "Training: Collective Statistic Normalization | Inference: Folded Erasure",
    candidates: [
      "Per-Channel BatchNorm",
      "Fused_SyncBatchNorm",
      "Folded_Erasure",
    ],
    chosen: {
      variant: "Training: Fused_SyncBatchNorm | Inference: Folded_Erasure",
      summary:
        "학습 시에는 채널별 통계 계산, 정규화, affine transform, running-state update가 collective path로 강하게 결합되고, 추론 시에는 선행 Conv/Linear와의 합성을 통해 standalone BatchNorm 노드를 제거할 수 있습니다.",
      reason: [
        "\\text{집단 통계 결합(Collective Statistic Coupling): } \\mu_B, \\sigma_B^2, \\gamma, \\beta, \\text{running stats} \\text{ 가 채널 기준으로 강하게 연결된다}",
        "\\text{학습 시 statistics reduction, normalization, affine transform, running-state update를 결합된 realization으로 유지할 수 있다}",
        "\\text{분산 학습에서는 global batch semantics 유지를 위해 synchronized statistics가 필요하다}",
        "\\text{추론 시 affine-normalization 식이 선행 Conv/Linear와 합성 가능하므로 } \\texttt{Folded\\_Erasure} \\text{ family가 성립한다}",
      ],
      applied_rewrites: [
        "Persistent CTA Reduction",
        "Sync Statistics Fusion",
        "Conv-BN Folding (Inference)",
      ],
    },
  },

  realizationSnapshot: {
    family:
      "Training: Collective Statistic Normalization | Inference: Folded Erasure",
    highlights: [
      "Per-channel collective reduction",
      "Running-state update fusion",
      "Cross-device statistic synchronization",
      "Inference-time Conv/Linear folding",
    ],
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
      inference:
        "Folded inference path can erase the standalone BatchNorm kernel.",
    },
  },
  
  normalizationFamily: {
    statScope: "batch-collective",
    statAxes: ["B", "H", "W"],
    affineMode: "channel-wise",
    runningState: true,
    inferenceFoldable: true,
    syncRequired: true,
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