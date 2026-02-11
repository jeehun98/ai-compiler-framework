export const gemmData = {
  id: "GEMM",
  category: "Linear Transform / Projection",

  canonical: {
    formula: "C = \\alpha(A \\times B) + \\beta C",
    shapes: {
      A: "M×K",
      B: "K×N",
      C: "M×N",
    },
    interpretation: {
      rowA: "sample i (A_i)",
      colB: "hypothesis j (B_j)",
      cij: "alignment score <A_i, B_j>",
    },
  },

  // 1) 의미론: '무엇을 보존해야 하는가'
  semantics: {
    thesis: "Semantic projection operator that evaluates relational hypotheses and merges state.",
    axes: {
      M: { name: "Samples", role: "batch of queries" },
      K: { name: "Hypothesis Search Space", role: "semantic channel / evidence accumulation" },
      N: { name: "Feature Channels", role: "projection outputs / logits" },
    },

    invariants: [
      {
        id: "INV_TOPK_ORDER",
        name: "Top-K / Rank Invariance",
        metric: "rowwise_argsort_preserve",
        threshold: "TopK@k=1..k0 preserved",
        applies_when: ["downstream=Softmax", "downstream=TopK", "beam_search_prelogits"],
        allows: ["low_bit_quant", "approx_dot", "early_exit"],
      },
      {
        id: "INV_SUBSPACE",
        name: "Subspace Projection Invariance",
        metric: "span_similarity(col(B), col(B'))",
        threshold: ">= 0.999 (cos-subspace)",
        allows: ["low_rank_factorization", "basis_compression"],
      },
      {
        id: "INV_BOUNDARY",
        name: "Decision Boundary Invariance",
        metric: "sign_consistency",
        threshold: ">= 99.99%",
        allows: ["aggressive_rewrite_if_verified"],
      },
    ],

    stateMerge: {
      enabled: true,
      meaning: "C_old as state, AB as observation; epilogue merges them.",
      params: { alpha: "\\alpha", beta: "\\beta", ratio: "\\alpha/\\beta" },
      state_types: ["residual", "optimizer_update", "running_stats"],
    },

    sensitivity: {
      downstream: [
        {
          name: "ReLU",
          rule: "if C_ij << 0 then precision can drop / early negative certainty",
          hint: "negative_region_low_priority",
        },
        {
          name: "Softmax",
          rule: "if (max - C_ij) large then exp -> 0, allow pruning/low precision",
          hint: "tail_prune_allowed",
        },
      ],
      tilePriority: "semantic_sparsity_predict",
    },
  },

  // 2) 허용 변형: '무엇을 바꿀 수 있는가'
  rewrites: {
    candidates: [
      {
        id: "RW_LOWRANK_B",
        name: "Low-Rank Factorization",
        transform: "B ≈ U V  =>  A(UV)",
        preconditions: ["energy_preserve >= 0.999", "subspace_invariant passes"],
        knobs: { rank_tolerance: 0.05, energy_preserve: 0.999 },
      },
      {
        id: "RW_SPARSE_PRUNE",
        name: "Sparse Pruning",
        transform: "mask(B_ij) if |B_ij| < eps",
        preconditions: ["semantic_density low", "order_invariant passes"],
        knobs: { sparsity_threshold: "eps", allow_sparse: true },
      },
      {
        id: "RW_EARLY_EXIT_K",
        name: "Incremental Accumulation (K early-exit)",
        transform: "stop K traversal when margin sufficient",
        preconditions: ["topk_margin >= margin0", "order_invariant passes"],
        knobs: { margin0: "profiled", confidence: "profiled" },
      },
      {
        id: "RW_ANCHOR_FUSION",
        name: "Semantic Anchor Fusion",
        transform: "GEMM + Bias + Norm + Act => Anchor",
        preconditions: ["same semantic unit", "no externally observed intermediate"],
      },
    ],
  },

  // 3) 비용함수: '무엇을 최소화하는가'
  costModel: {
    compute: ["FLOPs", "bandwidth", "occupancy"],
    semanticLoss: "λ1*RankLoss + λ2*OrderViolation + λ3*BoundaryDrift",
    weights_hint: {
      default: { RankLoss: 1.0, OrderViolation: 5.0, BoundaryDrift: 3.0 },
      safety_critical: { RankLoss: 5.0, OrderViolation: 20.0, BoundaryDrift: 20.0 },
    },
  },

  // 4) lowering 선택: '결국 어떤 커널을 택했는가' (페이지에서 제일 설득력 생김)
  lowering: {
    chosen: {
      variant: "TensorCore_GEMM_EpilogueFused",
      reason: [
        "downstream=Softmax => enforce Top-K order invariant",
        "profile shows tail prune safe in 78% tiles",
        "epilogue stateMerge enabled => avoid materializing C_intermediate",
      ],
    },
    options: [
      "Full GEMM",
      "Low-rank GEMM",
      "Sparse GEMM",
      "Tensor Core variant",
      "Fused epilogue variant",
    ],
  },

  // 5) 물리 최적화: '어떻게 빨라졌는가' (의미론과 연결해서 써야 함)
  kernel: {
    strategy: "2D Hierarchical Tiling",
    details: [
      { technique: "Shared-memory tiling", semantic_link: "reuse evidence across hypotheses (K-axis)" },
      { technique: "K-loop unroll", semantic_link: "accelerate hypothesis testing throughput" },
      { technique: "Epilogue fusion", semantic_link: "state merge is a semantic unit; avoid intermediate write" },
    ],
    metrics: { memory_reuse: "14.2x", throughput: "84.2 TFLOPS", occupancy: 92 },
  },

  performance: {
    latency: { ours: 120, pytorch: 210, torch_compile: 155 },
  },

  cudaCode: `// AICF Generated: Semantic-fused GEMM
__global__ void gemm_semantic_fused(...) {
  // Register blocking for K-axis (hypothesis search space)
  // Shared memory tiling (evidence reuse)
  // Epilogue fusion (state merge: α, β)
}`,
};
