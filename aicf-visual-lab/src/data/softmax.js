// src/data/softmax.js
export const softmaxData = {
  id: "Softmax",
  category: "Hypothesis Competition / Entropy-based Selector",

  canonical: {
    formula: "p_i = \\frac{e^{x_i}}{\\sum_j e^{x_j}}",
    shapes: {
      x: "M×N (row-wise softmax over N)",
      p: "M×N",
    },
    interpretation: {
      x: "logits / hypothesis scores (competition field)",
      p: "probability simplex (selection distribution)",
      "x_{max}": "dominant hypothesis score (winner candidate)",
    },
  },

  // 1) 의미론: '무엇을 보존해야 하는가'
  semantics: {
    thesis:
      "Softmax is a hypothesis competition system and entropy compressor: it silences weak hypotheses and amplifies strong ones under probabilistic contracts.",

    axes: {
      M: { name: "Rows", role: "tokens / queries / batch elements" },
      N: { name: "Hypotheses", role: "candidate set competing for survival" },
    },

    invariants: [
      {
        id: "INV_SUM_TO_ONE",
        name: "Sum-to-One Invariance",
        metric: "\\sum_i p_i",
        threshold: "= 1",
        applies_when: ["probabilistic_contract=true"],
        allows: ["sparse_truncation_with_renorm", "approx_exp_if_nonneg"],
      },
      {
        id: "INV_NONNEG",
        name: "Non-Negativity Contract",
        metric: "p_i",
        threshold: "\\ge 0",
        applies_when: ["probabilistic_contract=true"],
        allows: ["lut_exp", "integer_exp_approx (nonneg)"],
      },
      {
        id: "INV_KL",
        name: "KL Divergence Contract",
        metric: "D_{KL}(p\\,\\|\\,p')",
        threshold: "< \\tau_{KL}",
        applies_when: ["semantic_equivalence_check=true"],
        allows: ["sparse_softmax", "approx_exp", "hybrid_rowwise_lowering"],
      },
      {
        id: "INV_TOPK_STABILITY",
        name: "Top-K Stability Under Logit Perturbation",
        metric: "\\delta_{max} < \\min_{i\\in TopK,\\ j\\notin TopK} \\; margin_{ij}",
        threshold: "TopK preserved",
        applies_when: ["downstream=TopK", "beam_search", "selection_rigidity=high"],
        allows: ["fp16_logits", "approx_exp", "active_set_exp_only"],
      },
    ],

    stateMerge: {
      enabled: false,
      meaning:
        "Softmax does not merge states; it transforms a competition field into a probability simplex under probabilistic invariants.",
      params: {},
      state_types: ["competition", "selection"],
    },

    attributes: {
      // Saliency / sparsity
      active_threshold: "\\tau",
      tail_mass: "profiled",
      active_hypothesis_size: "|\\mathcal{H}_{active}| (profiled)",
      selection_rigidity: "low|high",

      // Numerical plasticity
      logit_noise_tolerance: "profiled",
      margin_sensitivity: "profiled",
      entropy_level: "profiled",
      selection_confidence: "profiled",

      // Training context
      gradient_clipping_necessity: "profiled",
      train_mode: "true|false",
    },

    sensitivity: {
      downstream: [
        {
          name: "Semantic Saliency Masking",
          rule:
            "If x_{max}-x_i \\gg 0 then p_i \\to 0 (semantic silence). Restrict exp to active set when tail_mass < \\epsilon under KL contract.",
          hint: "active_set_exp_only",
        },
        {
          name: "Logit Perturbation Tolerance",
          rule:
            "If logit_noise_tolerance is high and margin_sensitivity is low, allow FP16 logits and LUT/approx exp while preserving Top-K stability.",
          hint: "approx_exp_allowed",
        },
        {
          name: "Training Gradient Sensitivity",
          rule:
            "In low-entropy, high-confidence regime, gradients can be tiny; approximate ops may have limited training impact (verify-gated).",
          hint: "train_mode_relaxed_if_verified",
        },
      ],
      tilePriority: "entropy_tailmass_margin_predict",
    },
  },

  // 2) 허용 변형: '무엇을 바꿀 수 있는가'
  rewrites: {
    candidates: [
      {
        id: "RW_ACTIVE_SET_SPARSE",
        name: "Active Hypothesis Set (Sparse Softmax)",
        transform:
          "\\mathcal{H}_{active}=\\{i\\mid x_{max}-x_i<\\tau\\},\\ \\text{compute exp only on }\\mathcal{H}_{active}\\ \\text{then renorm}",
        preconditions: ["tail_mass < \\epsilon", "selection_rigidity=high", "KL contract holds"],
        knobs: { tau: "\\tau", epsilon: "\\epsilon", renorm: true },
      },
      {
        id: "RW_LUT_EXP",
        name: "LUT / Approx Exp (Nonneg-Guaranteed)",
        transform: "e^{x}\\ \\Rightarrow\\ \\widetilde{e^{x}}\\ \\text{(LUT/approx), enforce } \\widetilde{e^{x}}\\ge 0",
        preconditions: ["nonneg contract holds", "KL contract holds OR topk stability holds"],
        knobs: { method: "LUT|poly|int_exp", clamp_nonneg: true },
      },
      {
        id: "RW_LOGSUMEXP_APPROX",
        name: "Approx LogSumExp (Stable Normalizer)",
        transform: "\\log\\sum_j e^{x_j}\\ \\Rightarrow\\ \\widetilde{\\log\\sum e^x}",
        preconditions: ["distribution stable", "KL contract holds"],
        knobs: { method: "topk_sum|blockwise", max_kl: "\\tau_{KL}" },
      },
      {
        id: "RW_HYBRID_ROWWISE",
        name: "Hybrid Row-wise Lowering (Entropy-aware)",
        transform:
          "high-entropy rows: approx exp\\ ;\\ low-entropy rows: sparse/active-set\\ ;\\ (verify-gated for training)",
        preconditions: ["entropy_level profiled", "rowwise strategy enabled", "KL contract holds"],
        knobs: { entropy_th: "\\tau_H", tail_eps: "\\epsilon" },
      },
      {
        id: "RW_ANCHOR_ATTEND_MERGE",
        name: 'Semantic Anchor: "Attend & Merge"',
        transform: "\\mathrm{Softmax}(QK^T)@V\\ \\Rightarrow\\ \\text{single semantic unit lowering}",
        preconditions: ["pattern=Softmax(QK^T)@V", "no externally observed intermediate"],
        knobs: { fuse_with_v: true },
      },
      {
        id: "RW_LOW_ENTROPY_ARGMAX",
        name: "Low-Entropy Argmax Selection (Contracted)",
        transform: "entropy(row)<\\tau_H\\ \\Rightarrow\\ output \\approx V[\\arg\\max]",
        preconditions: ["entropy(row) < \\tau_H", "selection_confidence high", "soft_weighting_not_required", "verify_gated"],
        knobs: { tau_H: "\\tau_H", confidence: "c0", verify_gated: true },
      },
    ],
  },

  // 3) 비용함수: '무엇을 최소화하는가'
  costModel: {
    compute: ["exp_cost", "reduction_cost", "bandwidth"],
    semanticLoss:
      "\\lambda_1\\cdot KLDrift + \\lambda_2\\cdot TopKViolation + \\lambda_3\\cdot ProbContractViolation",
    weights_hint: {
      default: { KLDrift: 10.0, TopKViolation: 8.0, ProbContractViolation: 20.0 },
      safety_critical: { KLDrift: 25.0, TopKViolation: 20.0, ProbContractViolation: 40.0 },
    },
    semanticCompute:
      "Cost_{semantic} \\propto |\\mathcal{H}_{active}| \\ \\text{(effective hypothesis count)}",
  },

  // 4) lowering 선택: '결국 어떤 커널을 택했는가'
  lowering: {
    chosen: {
      variant: "HybridSoftmax_ActiveSet_LUTExp_Renorm",
      reason: [
        "tail_mass < \\epsilon => semantic silence in tail; restrict exp to active set",
        "selection_rigidity=high => preserve Top-K stability; verify KL drift bounded",
        "LUT exp used with nonneg guarantee to reduce exp cost",
        "sum-to-one enforced via renormalization after sparsification",
      ],
      applied_rewrites: ["RW_ACTIVE_SET_SPARSE", "RW_LUT_EXP", "RW_HYBRID_ROWWISE"],
    },
    options: [
      "FullSoftmax",
      "StableSoftmax(LogSumExp)",
      "SparseSoftmax(ActiveSet+Renorm)",
      "Softmax_LUTExp",
      "HybridSoftmax(Entropy-aware)",
      "Fused_AttendAndMerge(Softmax@V)",
    ],
  },

  // 5) 물리 최적화: '어떻게 빨라졌는가'
  kernel: {
    strategy: "Row-wise max-subtract + exp + sum + renorm (optionally sparse)",
    details: [
      { technique: "rowwise max subtraction", semantic_link: "stability; preserves margins for Top-K" },
      { technique: "active-set masking", semantic_link: "semantic silence: ignore tail hypotheses under KL contract" },
      { technique: "approx exp (LUT/poly)", semantic_link: "reduce exp cost while preserving nonneg + KL" },
      { technique: "renormalization", semantic_link: "enforce sum-to-one after truncation" },
      { technique: "anchor fusion (Softmax@V)", semantic_link: "Attend & Merge is single semantic unit; avoid intermediate write" },
    ],
    metrics: { memory_reuse: "—", throughput: "—", occupancy: "—" },
  },

  performance: {
    latency: { ours: "—", pytorch: "—", torch_compile: "—" },
  },

  cudaCode: `// AICF: Softmax (hypothesis competition, contracted sparsity/approx)
__global__ void softmax_rowwise(...) {
  // 1) rowwise max subtraction (stability)
  // 2) (optional) build active set: x_max - x_i < tau
  // 3) exp (LUT/approx) on active set (nonneg guarantee)
  // 4) sum + renorm (sum-to-one contract)
  // 5) optional anchor: fuse with @V (Attend & Merge)
}`,
};
