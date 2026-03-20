// src/data/theory/properties/tile_composable.js

const tileComposable = {
  id: "TileComposable",
  group: "structural",
  profileKey: "tile_composable",
  title: "Tile Composable",
  subtitle: "Partition Composition Property",
  hero: {
    lead:
      "A computation is tile-composable when partitioned subcomputations over tiles can be composed to reconstruct the same semantic result as the unpartitioned whole.",
    canonicalLatex:
      "F(X)=C\\big(F(X_1),F(X_2),\\dots,F(X_n)\\big)",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Partitionable Domain",
          v: "입력 또는 iteration domain이 유효한 타일 집합으로 분해 가능해야 한다.",
        },
        {
          k: "Tile-Level Semantics",
          v: "각 타일 위에서의 부분 계산이 전체 결과를 구성하는 데 의미 있는 기여를 해야 한다.",
        },
        {
          k: "Composable Reconstruction",
          v: "타일 부분 결과들은 결합 연산을 통해 원래 전체 계산과 동일한 의미를 복원할 수 있어야 한다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "전체 계산을 여러 tile-local computation과 combination 단계로 재표현할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "tiling, blocking, tile-local lowering, hierarchical execution decomposition을 정당화할 수 있다.",
        },
      ],
      latex:
        "F(X)=C\\big(F(X_1),F(X_2),\\dots,F(X_n)\\big)",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "boxes",
          title: "Valid Partition",
          desc: "도메인 분할은 전체 의미를 빠뜨리거나 중복 왜곡하지 않는 유효한 타일 구조여야 한다.",
        },
        {
          id: "02",
          icon: "merge",
          title: "Composable Partial Result",
          desc: "타일 부분 결과는 전체 결과를 복원하는 결합 규칙 아래에서 의미 있게 합성 가능해야 한다.",
        },
        {
          id: "03",
          icon: "shield",
          title: "Boundary Correctness",
          desc: "타일 경계, overlap, halo dependency가 있을 경우에도 전체 semantic result와 일치해야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Domain tiling",
        "Blockwise decomposition",
        "Hierarchical execution partitioning",
        "Tile-local lowering",
      ],
    },
    boundary: {
      items: [
        "타일 경계 밖의 dependency가 부분 결과로 충분히 요약되지 않으면 tile composition은 성립하지 않는다.",
        "partition이 전체 의미를 누락하거나 왜곡하면 transform은 불법이다.",
        "tile-local result만으로는 global coupling을 복원할 수 없는 계산에서는 이 property가 제한된다.",
      ],
    },
    relatedConstructions: {
      items: ["GEMMTiling", "BlockedConvolution", "BlockwiseAttention"],
    },
    relatedTransforms: {
      items: [
        "Loop tiling",
        "Blockwise lowering",
        "Hierarchical tile composition",
      ],
    },
  },
};

export default tileComposable;