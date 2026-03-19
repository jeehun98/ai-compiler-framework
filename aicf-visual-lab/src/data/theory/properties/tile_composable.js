const tileComposable = {
  id: "TileComposable",
  title: "Tile-Composable",
  subtitle: "Structure Property",
  hero: {
    lead:
      "A computation is tile-composable when it can be partitioned into local tiles and recomposed into the original global result.",
    canonicalLatex:
      "F(X) = Merge(\\{F_t(X_t)\\}_{t \\in T})",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "연산을 tile 단위로 나누어 계산하고 그 결과를 다시 합성할 수 있다.",
        },
        {
          k: "Requirement",
          v: "tile-local 결과가 전체 결과의 유효한 부분이어야 하고 합성 규칙이 존재해야 한다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "tiling, blockwise compute, shared-memory blocking의 핵심 근거다.",
        },
        {
          k: "Compiler View",
          v: "global computation을 local realization family로 낮출 수 있게 한다.",
        },
      ],
      latex: "F(X) = Merge(\\{F_t(X_t)\\})",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "boxes",
          title: "Local Validity",
          desc: "각 tile 계산이 전체 결과의 유효한 부분이어야 한다.",
          metric: "F(X)=Merge(\\{F_t(X_t)\\})",
        },
        {
          id: "02",
          icon: "merge",
          title: "Composable Merge",
          desc: "tile 간 결과가 의미적으로 합쳐질 수 있어야 한다.",
        },
        {
          id: "03",
          icon: "shield",
          title: "Boundary Safety",
          desc: "cross-tile dependency는 없거나 mergeable form이어야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Tile decomposition",
        "Shared-memory blocking",
        "Blockwise compute",
        "Tensor-core oriented mapping",
      ],
    },
    boundary: {
      items: [
        "global coupling이 local form으로 분해 불가한 경우",
        "tile boundary dependency를 merge할 수 없는 경우",
      ],
    },
    relatedOps: {
      items: ["GEMM", "Softmax", "LayerNorm"],
    },
    relatedTransforms: {
      items: [
        "Tiling",
        "Blockwise realization",
        "Tile-local accumulation",
      ],
    },
  },
};

export default tileComposable;