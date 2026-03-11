const onlineNormDetail = {
  overview: {
    title: "Method Overview",
    summary:
      "Online Reducible Norm converts mean and variance computation from multi-pass reduction into a single-pass streaming reduction.",
    problem:
      "Traditional normalization implementations compute mean and variance in separate passes, causing the input tensor to be read multiple times and increasing HBM bandwidth usage.",
    property:
      "The key observation is that statistics can be represented as a mergeable state. Welford's formulation maintains (count, mean, M2), which allows partial segments to be merged safely. This makes the statistic suitable for both streaming accumulation and parallel tile reductions.",
    impact:
      "By avoiding repeated reads of the same activation data, normalization kernels reduce intermediate traffic and alleviate memory bandwidth pressure.",
  },

  theory: {
    title: "Math & Logic",
    body: [
      "Mean and variance may appear to require full materialization of the vector before computation, but they can instead be expressed as a mergeable statistical state.",
      "Welford's algorithm maintains a running state consisting of count, mean, and M2, updating the statistics incrementally as new samples arrive.",
      "Because two partial Welford states can be merged safely, the algorithm supports both streaming updates and block-level parallel reductions.",
      "This means normalization statistics do not require full tensor materialization and can instead be computed through reducible state accumulation.",
    ],
    bullets: [
      "Reducer state: (count, mean, M2)",
      "Associative-style state merge",
      "Single-pass statistic accumulation",
      "Eliminates multi-pass HBM rereads",
    ],
  },

  hardware: {
    title: "Physical Analysis",
    body: [
      "From a hardware perspective, the primary advantage is that the activation tensor does not need to be read multiple times.",
      "Tile-level partial statistics can be accumulated in registers or shared memory, followed by a block reduction to finalize the normalization factor.",
      "Compared to implementations that scan the input multiple times, this approach significantly reduces global memory traffic.",
      "The benefit becomes more pronounced when kernels are memory-bandwidth bound.",
    ],
    bullets: [
      "Reduced HBM rereads",
      "Register / shared-memory accumulation",
      "Naturally compatible with reduction trees",
      "Foundation for fused normalization kernels",
    ],
  },

  compiler: {
    title: "MCIR Implementation",
    body: [
      "Within MCIR, this method should not be represented as separate mean and variance operators, but rather as a reducible-statistic property.",
      "The key legality condition is whether the statistic can be decomposed into a merge-safe state representation.",
      "During lowering, the computation can be structured as tile-local accumulation followed by hierarchical state merges.",
      "The final stage applies the normalization scale using the finalized statistics.",
    ],
    bullets: [
      "Property: reducible_state(statistics)",
      "Legality: merge-safe statistical state",
      "Rewrite: multi-pass norm → streaming stat reduction",
      "Kernel mapping: tile accumulation + hierarchical reduction",
    ],
  },
};

export default onlineNormDetail;