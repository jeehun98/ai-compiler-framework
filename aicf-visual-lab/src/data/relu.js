// src/data/relu.js
export const reluData = {
  id: "ReLU",
  category: "Nonlinear Gating / Half-Space Rectification Operator",

  canonical: {
    formula: "y_i = \\max(0, x_i)",
    shapes: {
      x: "… (elementwise)",
      y: "… (same shape as x)",
    },
    interpretation: {
      x: "pre-activation field (signed evidence)",
      y: "rectified evidence (positive-only, gated)",
      "y\\ge 0": "non-negativity contract",
    },
  },

  // 1) 의미론: '무엇을 보존해야 하는가'
  semantics: {
    thesis:
      "ReLU is a half-space rectifier: it folds negative space onto the zero hyperplane, producing semantic sparsity and introducing a hard decision boundary into a linear representation space.",

    axes: {
      N: { name: "Elements", role: "independent evidence units (gated elementwise)" },
      boundary: { name: "Zero Hyperplane", role: "decision boundary that splits active/inactive regions" },
    },

    invariants: [
      {
        id: "INV_NONNEG",
        name: "Non-Negativity Contract",
        metric: "y_i",
        threshold: "\\ge 0",
        applies_when: ["relu=true"],
        allows: ["unsigned_arithmetic_if_downstream_allows"],
      },
      {
        id: "INV_SIGN_BOUNDARY",
        name: "Boundary Preservation Contract (Sign)",
        metric: "\\mathrm{sign}(x_i) = \\mathrm{sign}(x_i')",
        threshold: "\\Rightarrow\\ \\mathrm{ReLU}(x_i)=\\mathrm{ReLU}(x_i')",
        applies_when: ["semantic_equivalence_check=true"],
        allows: ["low_precision_x (far_from_boundary)", "approx_preop_if_sign_preserved"],
      },
      {
        id: "INV_LINEARITY_ZONE",
        name: "Linearity-Zone Contract",
        metric: "x_i > \\delta",
        threshold: "\\Rightarrow\\ \\mathrm{ReLU}(x_i)=x_i",
        applies_when: ["positive_region_dominant=true"],
        allows: ["relu_elimination", "fuse_as_identity"],
      },
      {
        id: "INV_POSITIVE_ORDER",
        name: "Positive Half-Space Order Preservation",
        metric: "x_i > x_j > 0",
        threshold: "\\Rightarrow\\ y_i > y_j",
        applies_when: ["both_positive=true"],
        allows: ["rank_based_downstream_safe"],
      },
    ],

    stateMerge: {
      enabled: false,
      meaning:
        "ReLU is not a merge; it is a gating operator that collapses negative evidence into a single attractor state (0).",
      params: {},
      state_types: ["gating", "sparsity_generation"],
    },

    attributes: {
      activation_density: "profiled",
      clipping_ratio: "\\frac{\\#(x_i<0)}{N}",
      positive_tail_energy: "\\|x_{x>0}\\|_2",
      zero_threshold_proximity: "profiled",
      dead_unit_ratio: "profiled",
      permanent_sparsity_potential: "profiled",
      downstream_requires_sign: "true|false",
    },

    sensitivity: {
      downstream: [
        {
          name: "Near-Boundary Sensitivity",
          rule:
            "If |x_i| \\approx 0, tiny numeric error can flip activation. High zero_threshold_proximity => require higher precision / stricter guards.",
          hint: "boundary_precision_required",
        },
        {
          name: "Semantic Sparsity Generation",
          rule:
            "High clipping_ratio implies strong semantic sparsity; enable zero-skipping / sparse pathways if downstream supports.",
          hint: "zero_skipping_candidate",
        },
        {
          name: "Dead Neuron Semantics",
          rule:
            "If dead_unit_ratio persists across windows, treat as structural pruning potential, not mere zeros.",
          hint: "structural_prune_candidate",
        },
        {
          name: "Unsigned Arithmetic Opportunity",
          rule:
            "If downstream does not need sign information, exploit y>=0 to use unsigned/saturating arithmetic and drop sign-bit handling.",
          hint: "unsigned_path_allowed",
        },
      ],
      tilePriority: "boundary_proximity_and_sparsity_predict",
    },
  },

  // 2) 허용 변형: '무엇을 바꿀 수 있는가'
  rewrites: {
    candidates: [
      {
        id: "RW_DEADZONE_FUSION",
        name: "Semantic Dead-Zone Fusion (Predictive Execution)",
        transform:
          "Linear(Conv/GEMM)\\rightarrow\\mathrm{ReLU}\\ :\\ predict\\ negative-dominant\\ tiles\\Rightarrow\\ skip/cheap-accum",
        preconditions: ["clipping_ratio high", "confidence high", "no external intermediate"],
        knobs: { confidence_window: "W", negate_margin: "m", apply_zero_skip: true },
      },
      {
        id: "RW_STRUCTURAL_PRUNE",
        name: "Structural Pruning Candidate",
        transform: "dead_unit_ratio\\ high\\ (persistent)\\Rightarrow\\ remove\\ channels/neurons",
        preconditions: ["permanent_sparsity_potential high", "window_profile_confidence high"],
        knobs: { window: "W", persistence: "p0" },
      },
      {
        id: "RW_UNSIGNED_OPT",
        name: "Unsigned / Saturating Arithmetic Optimization",
        transform: "y\\ge 0\\Rightarrow\\ unsigned\\ or\\ saturating\\ downstream",
        preconditions: ["downstream_requires_sign=false"],
        knobs: { use_unsigned: true, saturating: true },
      },
      {
        id: "RW_RELU_ERASE",
        name: "ReLU Elimination (Semantic Erasure)",
        transform: "\\forall i,\\ x_i>\\delta\\Rightarrow\\ remove\\ ReLU",
        preconditions: ["linearity_zone_verified", "verify_mode=true OR profiled_confidence high"],
        knobs: { delta: "\\delta", verify_gated: true },
      },
      {
        id: "RW_FUSE_EPILOGUE",
        name: "Epilogue Fusion (Preferred)",
        transform: "ReLU\\Rightarrow\\ fuse\\ into\\ producer\\ epilogue",
        preconditions: ["producer supports epilogue", "no externally observed intermediate"],
      },
    ],
  },

  // 3) 비용함수: '무엇을 최소화하는가'
  costModel: {
    compute: ["elementwise_max", "bandwidth"],
    semanticLoss:
      "\\lambda_1\\cdot BoundaryFlip + \\lambda_2\\cdot SparsityPatternDrift + \\lambda_3\\cdot RankViolation(positive_space)",
    weights_hint: {
      default: { BoundaryFlip: 20.0, SparsityPatternDrift: 8.0, RankViolation: 4.0 },
      safety_critical: { BoundaryFlip: 50.0, SparsityPatternDrift: 20.0, RankViolation: 10.0 },
    },
    semanticCompute: "Cost_{semantic} \\propto \\text{boundary proximity density} + \\text{sparsity exploitation constraints}",
  },

  // 4) lowering 선택: '결국 어떤 커널을 택했는가'
  lowering: {
    chosen: {
      variant: "FusedEpilogue_ReLU_WithBoundaryGuard",
      reason: [
        "ReLU is a gating boundary; prefer producer epilogue fusion to avoid extra memory IO",
        "boundary proximity can flip semantics; add boundary guard for near-zero region",
        "high sparsity tiles enable optional zero-skipping under confidence contract",
      ],
      applied_rewrites: ["RW_FUSE_EPILOGUE"],
    },
    options: [
      "StandaloneReLU",
      "FusedEpilogue_ReLU",
      "FusedEpilogue_ReLU_WithBoundaryGuard",
      "ReLU_UnsignedPath (downstream-allowed)",
      "ReLU_Erased (verified linearity zone)",
    ],
  },

  // 5) 물리 최적화: '어떻게 빨라졌는가'
  kernel: {
    strategy: "Elementwise max with optional boundary guard; prefer fusion",
    details: [
      { technique: "epilogue fusion", semantic_link: "gating is part of producer semantic unit; avoid intermediate write" },
      { technique: "vectorized load/store", semantic_link: "bandwidth-bound; maximize throughput on contiguous tensors" },
      { technique: "boundary guard fastpath", semantic_link: "protect sign flips near zero while keeping hot path fast" },
      { technique: "zero-skipping (optional)", semantic_link: "semantic sparsity allows skipping under contracted confidence" },
    ],
    metrics: { memory_reuse: "N/A (Streaming)", throughput: "High", occupancy: "—" },
  },

  performance: {
    latency: { ours: "—", pytorch: "—", torch_compile: "—" },
  },

  cudaCode: `// AICF: ReLU (half-space rectification)
__global__ void relu(...) {
  // Hot path: y = max(0, x)
  // Optional: boundary-guard for |x| ~ 0 (prevent semantic flip)
  // Preferred: fuse into producer epilogue (GEMM/Conv/BiasAdd/Norm)
}`,
};
