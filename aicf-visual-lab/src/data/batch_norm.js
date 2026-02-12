// src/data/batch_norm.js
export const batchNormData = {
  id: "BatchNorm",
  category: "Cross-Sample Statistical Alignment / Collective Distribution Contract Operator",

  canonical: {
    formula: [
      "\\mu_c = \\frac{1}{B}\\sum_{b=1}^{B} x_{b,c}",
      "\\sigma_c^2 = \\frac{1}{B}\\sum_{b=1}^{B} (x_{b,c}-\\mu_c)^2",
      "\\hat{x}_{b,c} = \\frac{x_{b,c}-\\mu_c}{\\sqrt{\\sigma_c^2+\\epsilon}}",
      "y_{b,c}=\\gamma_c\\hat{x}_{b,c}+\\beta_c",
      "\\text{(inference)}\\quad y = \\gamma\\frac{x-\\hat{\\mu}}{\\sqrt{\\hat{\\sigma}^2+\\epsilon}}+\\beta",
    ].join("\\\\"),
    shapes: {
      x: "B×C×… (reduce over B for each channel C)",
      "\\gamma": "C",
      "\\beta": "C",
      running_mean: "C",
      running_var: "C",
      y: "B×C×…",
    },
    interpretation: {
      x: "activations coupled across samples (collective field)",
      "\\mu_c,\\sigma_c^2": "batch statistics per channel (collective reference frame)",
      "\\hat{\\mu},\\hat{\\sigma}^2": "running statistics (inference reference frame)",
      "\\gamma,\\beta": "affine calibration per channel",
      y: "aligned representation under a collective distribution contract",
    },
  },

  // 1) 의미론: '무엇을 보존해야 하는가'
  semantics: {
    thesis:
      "BatchNorm is a collective distribution contract operator: it enforces channel-wise alignment using cross-sample statistics, coupling samples in training and bridging training-to-inference via running stats.",

    axes: {
      B: { name: "Batch Axis", role: "collective sample set for statistics" },
      C: { name: "Channels", role: "independent per-channel contracts" },
    },

    invariants: [
      {
        id: "INV_CHANNEL_DIST",
        name: "Channel Distribution Contract",
        metric: "\\mathbb{E}[y_c],\\ \\mathrm{Var}(y_c)",
        threshold: "\\mathbb{E}[y_c]\\approx 0,\\ \\mathrm{Var}(y_c)\\approx 1\\ \\pm\\ (\\delta_\\mu,\\delta_\\sigma)",
        applies_when: ["mode=training", "collective_contract=true"],
        allows: ["approx_rsqrt (only if stable)", "vectorized_affine"],
      },
      {
        id: "INV_INTERCHANNEL_INDEP",
        name: "Inter-Channel Independence",
        metric: "\\mathrm{Cov}(y_{c1}, y_{c2})",
        threshold: "\\approx 0\\ \\text{(no cross-channel leakage)}",
        applies_when: ["mode=training", "per_channel_independence=true"],
        allows: ["channelwise_parallel_reduce"],
      },
      {
        id: "INV_WITHIN_CHANNEL_RANK",
        name: "Rank Preservation within Channel",
        metric: "\\Delta\\mathrm{rank}(y_{:,c})",
        threshold: "\\le \\tau_{rank}",
        applies_when: ["mode=training", "downstream=Ranking|TopK (per-channel)"],
        allows: ["reduced_precision_affine_if_stable"],
      },
      {
        id: "INV_TRAIN_INFER_DIVERGENCE",
        name: "Training–Inference Divergence Contract",
        metric: "D = \\|\\mu_{batch}-\\hat{\\mu}\\| + \\|\\sigma_{batch}-\\hat{\\sigma}\\|",
        threshold: "D < \\tau",
        applies_when: ["mode=inference", "running_stats_available=true"],
        allows: ["inference_folding", "erase_bn_node"],
      },
    ],

    stateMerge: {
      enabled: true,
      meaning:
        "In training, BatchNorm performs a collective stateful transform: batch stats define the reference frame and running stats are updated as a bridge to inference.",
      params: { eps: "\\epsilon", momentum: "m" },
      state_types: ["running_mean", "running_var", "collective_reference_frame"],
    },

    attributes: {
      mode: "training|inference",
      cross_sample_coupling: true,

      // running state
      running_mean: "\\hat{\\mu}",
      running_var: "\\hat{\\sigma}^2",
      momentum: "m",

      // stability & risk
      stat_stability_index: "profiled",
      outlier_influence_factor: "profiled",
      running_stat_drift: "profiled",
      batch_effective_sample_size: "profiled",

      // multi-device coupling
      sync_batchnorm: "true|false",
      allreduce_cost: "profiled",
    },

    sensitivity: {
      downstream: [
        {
          name: "Cross-Sample Coupling Leakage",
          rule:
            "If a sample dominates channel statistics (|x_{b,c}| \\gg others), it can distort \\mu_c,\\sigma_c^2 and shift other samples' representations (semantic coupling leakage).",
          hint: "outlier_influence_guard",
        },
        {
          name: "Batch Size Crisis",
          rule:
            "If batch_effective_sample_size is small, statistics become noisy and contract becomes unstable; suggest semantic alternatives (GroupNorm/LayerNorm/InstanceNorm) instead of silent rewrite.",
          hint: "norm_alternative_candidate",
        },
        {
          name: "Train–Infer Drift",
          rule:
            "If D is large, inference stability is not guaranteed; treat running_stat_drift as high and disable aggressive approximations/folding.",
          hint: "disable_fold_when_drift_high",
        },
      ],
      tilePriority: "stat_stability_and_outlier_predict",
    },
  },

  // 2) 허용 변형: '무엇을 바꿀 수 있는가'
  rewrites: {
    candidates: [
      {
        id: "RW_INFER_FOLD_ERASE",
        name: "Inference Folding (Semantic Erasure)",
        transform:
          "y = ax + b\\ \\Rightarrow\\ fold\\ into\\ preceding\\ Conv/Linear\\ (erase\\ BN\\ node)",
        preconditions: ["mode=inference", "D<\\tau", "running_stat_drift low", "fusion boundary allowed"],
        knobs: { erase_node: true },
      },
      {
        id: "RW_SYNC_STATS",
        name: "Sync Statistics (Multi-GPU Contract)",
        transform: "\\mu_c,\\sigma_c^2\\ \\Rightarrow\\ AllReduce\\ synced\\ stats",
        preconditions: ["sync_batchnorm=true", "collective_contract=true"],
        knobs: { allreduce: true },
      },
      {
        id: "RW_NO_APPROX_UNSTABLE",
        name: "Disable Approx Under Instability",
        transform: "approx_rsqrt/off\\ ,\\ high-precision stats/on",
        preconditions: ["stat_stability_index low OR outlier_influence_factor high"],
        knobs: { rsqrt_mode: "precise", stats_dtype: "fp32" },
      },
      {
        id: "RW_NORM_ALTERNATIVE_HINT",
        name: "Semantic Alternative Suggestion (Not Auto-Rewrite)",
        transform: "BatchNorm \\Rightarrow {GroupNorm, LayerNorm, InstanceNorm} (proposal)",
        preconditions: ["batch_size_crisis", "contract_unstable"],
        knobs: { proposal_only: true },
      },
    ],
  },

  // 3) 비용함수: '무엇을 최소화하는가'
  costModel: {
    compute: ["batch_reduce_cost", "affine_cost", "bandwidth"],
    semanticLoss:
      "\\lambda_1\\cdot DistContractViolation + \\lambda_2\\cdot CouplingLeakage + \\lambda_3\\cdot TrainInferDrift + \\lambda_4\\cdot SyncOverhead",
    weights_hint: {
      default: { DistContractViolation: 10.0, CouplingLeakage: 12.0, TrainInferDrift: 15.0, SyncOverhead: 6.0 },
      safety_critical: { DistContractViolation: 25.0, CouplingLeakage: 30.0, TrainInferDrift: 40.0, SyncOverhead: 12.0 },
    },
    semanticCompute: "Cost_{semantic} \\propto \\text{stats synchronization}(AllReduce) + \\text{running-state upkeep}",
  },

  // 4) lowering 선택: '결국 어떤 커널을 택했는가'
  lowering: {
    chosen: {
      variant: "Training: SyncBatchNorm_StableStats | Inference: FoldedBN_Erased",
      reason: [
        "training mode => cross-sample coupling is semantic core; enforce collective contract with stable fp32 stats",
        "multi-GPU => sync stats is required to maintain consistent reference frame",
        "inference mode => BN reduces to linear transform; fold into preceding layer as semantic erasure when drift is low",
      ],
      applied_rewrites: ["RW_SYNC_STATS", "RW_INFER_FOLD_ERASE"],
    },
    options: [
      "BatchNorm_Training_LocalStats",
      "SyncBatchNorm_Training_AllReduce",
      "BatchNorm_Inference_RunningStats",
      "FoldedBN_Erased (Conv/Linear fused)",
    ],
  },

  // 5) 물리 최적화: '어떻게 빨라졌는가'
  kernel: {
    strategy: "Channel-wise batch reduction + normalize + affine; optional AllReduce for SyncBN",
    details: [
      { technique: "two-pass reduction (mean/var)", semantic_link: "enforce collective reference frame per channel" },
      { technique: "vectorized normalize+affine", semantic_link: "apply contract efficiently; minimize bandwidth" },
      { technique: "fp32 stats accumulation", semantic_link: "protect contract under small batch/outliers" },
      { technique: "AllReduce stats (SyncBN)", semantic_link: "maintain global collective frame across devices" },
      { technique: "inference folding", semantic_link: "semantic erasure: remove node and memory round-trips" },
    ],
    metrics: { memory_reuse: "—", throughput: "—", occupancy: "—" },
  },

  performance: {
    latency: { ours: "—", pytorch: "—", torch_compile: "—" },
  },

  cudaCode: `// AICF: BatchNorm (collective distribution contract)
__global__ void batch_norm_train(...) {
  // 1) channel-wise mean/var reduction over batch (fp32 stats)
  // 2) (optional) AllReduce stats for SyncBN (multi-GPU)
  // 3) normalize + affine (gamma/beta)
}

__global__ void batch_norm_infer_or_folded(...) {
  // inference uses running stats
  // preferred: fold into preceding Conv/Linear (erase BN node)
}`,
};
