const tiledExecutionEquivalence = {
  id: "TiledExecutionEquivalence",
  profileKey: "tiled_execution_equivalence",
  group: "execution-meaning",
  title: "Tiled Execution Equivalence",
  subtitle: "Partitioned execution must reconstruct the same whole-operation meaning",

  hero: {
    lead:
      "Large operators may be split into tiles, blocks, or stages only when the recomposed result preserves the original whole-operation meaning.",
    canonicalLatex:
      "f(X) = \\mathrm{compose}(f(T_1), f(T_2), \\dots, f(T_n))",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Partition-Safe Meaning",
          v: "Each tile must compute a valid local contribution to the original operation.",
        },
        {
          k: "Correct Recomposition",
          v: "The final recomposition step must reconstruct the same global result.",
        },
        {
          k: "Boundary Completeness",
          v: "No tile boundary may introduce omission, duplication, or inconsistent state.",
        },
      ],
      latex: "Y = \\mathrm{compose}(Y_1, Y_2, \\dots, Y_n)",
      preview: [
        {
          k: "Compiler View",
          v: "Tiling is legal only when the operation admits local computation plus sound global recomposition.",
        },
        {
          k: "Kernel View",
          v: "Tile-local accumulators, staging buffers, and cross-block merge must match the original operator meaning.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "boxes",
          title: "Local Contribution Validity",
          desc:
            "Each tile result must represent a valid partial contribution of the original operator.",
          metric: "T_i \\Rightarrow Y_i \\subseteq Y_{global}",
        },
        {
          id: "02",
          icon: "gitmerge",
          title: "Recomposition Correctness",
          desc:
            "Tile outputs must be recombined by a rule that preserves the original global operation.",
          metric: "Y=\\mathrm{compose}(Y_1,\\dots,Y_n)",
        },
        {
          id: "03",
          icon: "layers",
          title: "Boundary Integrity",
          desc:
            "Cross-tile interaction must not be lost or misapplied at partition boundaries.",
          metric: "\\text{boundary-safe partition}",
        },
      ],
    },

    preserves: {
      items: [
        "Whole-operation meaning under partition",
        "Boundary-safe recomposition",
        "Tile-local correctness",
        "Global equivalence after merge",
      ],
    },

    failure: {
      items: [
        "cross-tile dependency is omitted or double-counted",
        "tile-local result is not sufficient for correct global recomposition",
        "partition boundary changes the effective semantics of the operation",
      ],
    },

    downstreamImpact: {
      items: [
        {
          k: "Matrix/Attention Kernels",
          v: "Incorrect tiling directly changes score, accumulation, or output composition.",
        },
        {
          k: "Fused Pipelines",
          v: "Tile mismatch can corrupt later fused epilogue or reduction stages.",
        },
      ],
    },

    relatedConstructions: {
      items: [
        { op: "gemm", label: "GEMM" },
        { op: "softmax", label: "Blockwise Softmax" },
        { op: "batch_norm", label: "Partitioned Statistics" },
      ],
    },

    relatedTransforms: {
      items: [
        "Block tiling",
        "Warp-level partial accumulation",
        "Stage-wise pipeline execution",
        "Tile-local compute plus global merge",
      ],
    },
  },
};

export default tiledExecutionEquivalence;