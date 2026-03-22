const reductionEquivalence = {
  id: "ReductionEquivalence",
  profileKey: "reduction_equivalence",
  group: "execution-meaning",
  title: "Reduction Equivalence",
  subtitle: "Merged partial states must preserve the same aggregate meaning",

  hero: {
    lead:
      "Sequential reduction, tree merge, blockwise accumulation, and multi-stage aggregation are allowed only when the final reduction meaning remains equivalent.",
    canonicalLatex:
      "R(X) = \\mathrm{merge}(R(X_1), R(X_2), \\dots, R(X_n))",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Same Aggregate Meaning",
          v: "Partial accumulation and later merge must produce the same aggregate interpretation as full reduction.",
        },
        {
          k: "Composable Partial State",
          v: "Each local state must carry enough information to reconstruct the global result correctly.",
        },
        {
          k: "Order-Tolerant Realization",
          v: "Execution order may change, but reduction meaning must not collapse into a different statistic.",
        },
      ],
      latex: "R_{global} = \\mathrm{merge}(R_1, R_2, \\dots, R_n)",
      preview: [
        {
          k: "Compiler View",
          v: "Reduction may be split across tiles, warps, or stages if mergeability is preserved.",
        },
        {
          k: "Kernel View",
          v: "Local partial sums, statistics, or weighted states must be recombined with the original mathematical rule.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "gitmerge",
          title: "Mergeable State",
          desc:
            "Local partial results must be representable as a state that supports correct global merging.",
          metric: "S_{global}=\\mathrm{merge}(S_1,S_2,\\dots,S_n)",
        },
        {
          id: "02",
          icon: "layers",
          title: "Boundary Completeness",
          desc:
            "Tile or chunk boundaries must not drop or duplicate contribution during recomposition.",
          metric: "\\sum_i X_i \\Rightarrow \\text{no missing region}",
        },
        {
          id: "03",
          icon: "shield",
          title: "Consistent Interpretation",
          desc:
            "Merged state must still represent the same reduction target rather than an approximation with different semantics.",
          metric: "\\hat{R}(X) \\equiv R(X)",
        },
      ],
    },

    preserves: {
      items: [
        "Aggregate reduction meaning",
        "Merge correctness across partitions",
        "Boundary-safe global recomposition",
        "Equivalent statistic interpretation",
      ],
    },

    failure: {
      items: [
        "tile boundary handling drops contribution from part of the input",
        "online merge state is insufficient to reconstruct the original statistic",
        "reordered merge changes the interpretation of the reduced result",
      ],
    },

    downstreamImpact: {
      items: [
        {
          k: "Attention / Weighted Sum",
          v: "Incorrect merge breaks attention accumulation and later score interpretation.",
        },
        {
          k: "Normalization",
          v: "Reduction mismatch corrupts denominator or statistic reconstruction.",
        },
      ],
    },

    relatedConstructions: {
      items: [
        { op: "softmax", label: "Softmax Denominator" },
        { op: "layer_norm", label: "LayerNorm Statistics" },
        { op: "batch_norm", label: "BatchNorm Reduction" },
        { op: "gemm", label: "Accumulation" },
      ],
    },

    relatedTransforms: {
      items: [
        "Tree reduction",
        "Warp/block split reduction",
        "Welford-style merge",
        "Blockwise weighted accumulation",
      ],
    },
  },
};

export default reductionEquivalence;