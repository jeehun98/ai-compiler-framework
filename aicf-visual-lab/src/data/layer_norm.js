// src/data/layer_norm.js
export const layerNormData = {
  id: "LayerNorm",
  category: "Distribution Reparameterization / Representation QA",

  canonical: {
    formula:
      "y = \\gamma\\,\\hat{x} + \\beta,\\quad \\hat{x}=\\frac{x-\\mu}{\\sqrt{\\sigma^2+\\epsilon}}",
    shapes: {
      x: "M×N (normalize over N)",
      "\\gamma": "1×N",
      "\\beta": "1×N",
      y: "M×N",
    },
    interpretation: {
      x: "pre-normalization representation (raw feature field)",
      "\\mu,\\sigma^2": "row-wise statistics (distribution parameters)",
      "\\epsilon": "amplification floor (semantic lower bound on gain)",
      "\\gamma,\\beta": "affine reparameterization (calibration after normalization)",
      y: "reparameterized representation (QA-stabilized)",
    },
  },

  // 1) 의미론: '무엇을 보존해야 하는가'
  semantics: {
    thesis:
      "LayerNorm is a distribution reparameterization and representation QA operator: it removes absolute energy while preserving relative ratio structure and stabilizing downstream behavior.",

    axes: {
      M: { name: "Samples", role: "rows / tokens / batch elements" },
      N: { name: "Feature Axis", role: "normalized dimension (distribution to be reparameterized)" },
    },

    invariants: [
      {
        id: "INV_RATIO_PRESERVE",
        name: "Semantic Ratio Preservation",
        metric: "\\rho_{ij}=\\frac{x_i-\\mu}{x_j-\\mu}",
        threshold: "|\\Delta\\rho| \\le \\tau_{\\rho}",
        applies_when: ["downstream!=strict_value_sensitive"],
        allows: ["topk_stats_approx", "small_component_mask", "fast_rsqrt"],
      },
      {
        id: "INV_DIST_CONTRACT",
        name: "Distribution Contract (Quantified)",
        metric: "\\mathbb{E}[y]\\in[-\\delta_\\mu,\\delta_\\mu],\\ \\mathrm{Var}(y)\\in[1-\\delta_\\sigma,1+\\delta_\\sigma]",
        threshold: "\\delta_\\mu\\le 10^{-2},\\ \\delta_\\sigma\\le 5\\times 10^{-2}",
        applies_when: ["verify_mode=false", "tolerance_contract_enabled=true"],
        allows: ["approx_rsqrt", "lut_rsqrt", "reduced_precision_stats"],
      },
      {
        id: "INV_DOWNSTREAM_STABILITY",
        name: "Downstream Preservation Constraint",
        metric: "D_{KL}(score_{orig}||score_{opt})",
        threshold: "D_{KL} \\le \\kappa",
        applies_when: ["downstream=Attention", "downstream=TopK/Ranking"],
        allows: ["aggressive_ln_rewrite_if_verified"],
      },
    ],

    // LN은 상태 병합이라기보단 "품질 보증 + 재좌표화"
    stateMerge: {
      enabled: false,
      meaning: "LayerNorm is not a merge; it is a QA reparameterization step that stabilizes representation geometry.",
      params: { epsilon: "\\epsilon" },
      state_types: ["distribution_QA", "reparameterization"],
    },

    attributes: {
      // Ratio dominance / heavy-tail
      dominant_component_ratio: "profiled",
      ratio_dominance_threshold: "\\tau",
      allow_topk_stats: "contracted",

      // Epsilon semantics
      noise_amplification_risk: "profiled",
      information_floor_sensitivity: "profiled",
      epsilon_threshold: "\\epsilon_{th}",

      // Quantified tolerance
      mean_error_tolerance: "1e-2",
      scale_error_tolerance: "5e-2",
      tolerance_contract_enabled: true,
    },

    sensitivity: {
      downstream: [
        {
          name: "Ratio-Dominant Rows (Heavy-tail / Sparse dominance)",
          rule:
            "If dominant_component_ratio > \\tau and downstream is not strict-value-sensitive, approximate stats using Top-K components; mask tiny components in variance accumulation.",
          hint: "topk_stats_approx_allowed",
        },
        {
          name: "Low-Variance Regime (\\sigma^2 \\ll \\epsilon)",
          rule:
            "Noise amplification risk: small noise can be over-amplified by epsilon floor. Apply semantic clipping guard when downstream is tolerant.",
          hint: "semantic_clipping_candidate",
        },
        {
          name: "Attention Stability",
          rule:
            "When downstream is attention, enforce distribution contract and bound KL drift of attention scores.",
          hint: "kl_guard_required",
        },
      ],
      tilePriority: "ratio_dominance_and_variance_predict",
    },
  },

  // 2) 허용 변형: '무엇을 바꿀 수 있는가'
  rewrites: {
    candidates: [
      {
        id: "RW_TOPK_STATS",
        name: "Top-K Statistics Approximation (Contracted)",
        transform:
          "\\mu,\\sigma^2\\ \\text{computed from Top-K components}\\ \\Rightarrow\\ \\hat{x}_{approx}",
        preconditions: ["dominant_component_ratio > \\tau", "downstream!=strict_value_sensitive", "ratio contract holds"],
        knobs: { topk: "k", tau: "\\tau", mask_small: true },
      },
      {
        id: "RW_FAST_RSQRT",
        name: "Fast rsqrt Approximation",
        transform:
          "\\mathrm{rsqrt}(\\sigma^2+\\epsilon)\\ \\Rightarrow\\ \\mathrm{rsqrt}_{approx}(\\cdot)",
        preconditions: ["distribution contract holds"],
        knobs: { method: "NR1|NR2|LUT", max_mean_err: "1e-2", max_scale_err: "5e-2" },
      },
      {
        id: "RW_SEMANTIC_CLIP",
        name: "Semantic Clipping / Identity Substitute (Guarded)",
        transform:
          "\\sigma^2 < \\epsilon_{th}\\ \\Rightarrow\\ y := \\gamma\\cdot 0 + \\beta\\ \\text{(or identity, contracted)}",
        preconditions: ["variance < \\epsilon_{th}", "downstream=tolerant", "train_mode=false OR verify_gated"],
        knobs: { epsilon_th: "\\epsilon_{th}", mode: "const_stabilize|identity" },
      },
      {
        id: "RW_NORM_PROJECT_FUSION",
        name: "Normalize → Project Fusion (Anchor Expansion)",
        transform:
          "LN(x)=\\gamma\\hat{x}+\\beta,\\ GEMM(W,\\cdot)\\Rightarrow W\\,\\mathrm{diag}(\\gamma)\\hat{x} + W\\beta",
        preconditions: ["pattern=LayerNorm→GEMM", "distribution contract holds", "fusion boundary allowed"],
        knobs: { fold_gamma: true, fold_beta: true },
      },
    ],
  },

  // 3) 비용함수: '무엇을 최소화하는가'
  costModel: {
    compute: ["reduction_cost", "rsqrt_cost", "bandwidth"],
    semanticLoss:
      "\\lambda_1\\cdot RatioViolation + \\lambda_2\\cdot DistContractViolation + \\lambda_3\\cdot KLDrift + \\lambda_4\\cdot AmplificationRisk",
    weights_hint: {
      default: { RatioViolation: 4.0, DistContractViolation: 8.0, KLDrift: 10.0, AmplificationRisk: 6.0 },
      safety_critical: { RatioViolation: 10.0, DistContractViolation: 25.0, KLDrift: 30.0, AmplificationRisk: 20.0 },
    },
  },

  // 4) lowering 선택: '결국 어떤 커널을 택했는가'
  lowering: {
    chosen: {
      variant: "Fused_LayerNorm_Vectorized_RsqrtApprox",
      reason: [
        "LayerNorm is representation QA => enforce quantified distribution contract",
        "rsqrt dominates runtime => allow LUT/NR approx under tolerance bounds",
        "ratio-dominant rows detected => enable Top-K stats only when contracted",
        "pattern LN→GEMM => enable anchor fusion when fusion boundary allows",
      ],
      applied_rewrites: ["RW_FAST_RSQRT"],
    },
    options: [
      "FullPrecision_LayerNorm",
      "Vectorized_LayerNorm",
      "LayerNorm_RsqrtApprox",
      "LayerNorm_TopKStatsApprox",
      "Fused_LayerNorm_GEMM (Anchor Fusion)",
    ],
  },

  // 5) 물리 최적화: '어떻게 빨라졌는가'
  kernel: {
    strategy: "Two-pass reduction (mean/var) + fused affine, vectorized IO",
    details: [
      { technique: "warp/block reduction", semantic_link: "distribution parameters must meet tolerance contract" },
      { technique: "fast rsqrt (NR/LUT)", semantic_link: "rsqrt cost traded under quantified distribution contract" },
      { technique: "vectorized load/store", semantic_link: "QA step is bandwidth-sensitive; maximize streaming" },
      { technique: "anchor fusion (LN→GEMM)", semantic_link: "absorb γ into W and β into bias to avoid intermediate write" },
    ],
    metrics: { memory_reuse: "—", throughput: "—", occupancy: "—" },
  },

  performance: {
    latency: { ours: "—", pytorch: "—", torch_compile: "—" },
  },

  cudaCode: `// AICF: LayerNorm (distribution QA, contracted approximations)
__global__ void layer_norm(...) {
  // 1) mean/var reduction (contracted tolerances)
  // 2) rsqrt approx (NR/LUT) under DistContract
  // 3) normalize + affine (gamma/beta)
  // 4) optional: anchor fusion path (LN -> GEMM)
}`,
};
