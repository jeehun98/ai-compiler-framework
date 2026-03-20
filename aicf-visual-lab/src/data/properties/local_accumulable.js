// src/data/theory/properties/local_accumulable.js

const localAccumulable = {
  id: "LocalAccumulable",
  group: "structural",
  profileKey: "local_accumulable",
  title: "Local Accumulable",
  subtitle: "Local Summary Property",
  hero: {
    lead:
      "A computation is local-accumulable when a global result can be constructed from locally accumulated summaries without requiring full immediate materialization of all intermediate interactions.",
    canonicalLatex:
      "F(X)=C\\big(A(X_1),A(X_2),\\dots,A(X_n)\\big)",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Local Summary",
          v: "입력의 각 부분은 전체 결과를 구성하기 위한 유효한 local state로 축약될 수 있어야 한다.",
        },
        {
          k: "Composable Combination",
          v: "local state들은 후속 결합 연산을 통해 전체 결과로 연결 가능해야 한다.",
        },
        {
          k: "Deferred Globalization",
          v: "전역 결과는 모든 세부 상호작용을 즉시 물질화하지 않고도 단계적으로 구성될 수 있어야 한다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "전체 계산을 local accumulation과 later combination 구조로 분해할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "blockwise accumulation, staged aggregation, on-chip partial state construction을 정당화할 수 있다.",
        },
      ],
      latex:
        "F(X)=C\\big(A(X_1),A(X_2),\\dots,A(X_n)\\big)",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "boxes",
          title: "Valid Local Summary",
          desc: "각 지역 부분계산은 전체 의미를 복원하는 데 필요한 충분한 partial state를 생성해야 한다.",
        },
        {
          id: "02",
          icon: "merge",
          title: "Composable Combination",
          desc: "local state들을 결합하는 절차가 전체 semantic result와 일치해야 한다.",
          metric: "F(X)=C\\big(A(X_1),\\dots,A(X_n)\\big)",
        },
        {
          id: "03",
          icon: "shield",
          title: "No Lost Dependency",
          desc: "local accumulation 과정에서 이후 결과에 필요한 핵심 의존성이 소실되면 안 된다.",
        },
      ],
    },
    enables: {
      items: [
        "Blockwise accumulation",
        "Staged aggregation",
        "On-chip partial state construction",
        "Deferred global writeback",
      ],
    },
    boundary: {
      items: [
        "전체 상호작용을 즉시 물질화해야만 의미를 유지할 수 있다면 local accumulation은 성립하지 않는다.",
        "local summary가 이후 결합에 필요한 정보를 충분히 보존하지 못하면 불법이다.",
        "cross-region dependency가 local state로 요약되지 않으면 staged accumulation은 정당화되지 않는다.",
      ],
    },
    relatedConstructions: {
      items: ["BlockwiseAttention", "OnlineSoftmaxState", "TileLocalReduction"],
    },
    relatedTransforms: {
      items: [
        "Local partial accumulation",
        "Blockwise state staging",
        "Late materialization of global result",
      ],
    },
  },
};

export default localAccumulable;