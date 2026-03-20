// src/data/theory/properties/associative_merge.js

const associativeMerge = {
  id: "AssociativeMerge",
  profileKey: "associative_merge",
  group: "foundational",
  title: "Associative Merge",
  subtitle: "Algebraic Merge Property",
  hero: {
    lead:
      "A merge structure is associative when valid partial states can be regrouped under different bracketings without changing the semantic result.",
    canonicalLatex:
      "M : S \\times S \\to S,\\qquad M(M(a,b),c)=M(a,M(b,c))\\quad \\forall a,b,c\\in S",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "State Space",
          v: "유효한 partial result들이 속하는 상태 공간 S가 정의되어야 한다.",
        },
        {
          k: "Closed Merge",
          v: "merge 연산 M은 S 위에 닫혀 있어야 하며, partial state를 다시 같은 종류의 state로 합쳐야 한다.",
        },
        {
          k: "Associativity",
          v: "임의의 a, b, c ∈ S 에 대해 묶는 순서가 달라도 merge 결과가 같아야 한다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "전체 집계를 여러 partial aggregation으로 분해한 뒤 다시 결합해도 동일한 의미를 유지할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "전역 aggregation을 local summary와 hierarchical merge tree 형태로 재구성할 수 있다.",
        },
      ],
      latex:
        "M : S \\times S \\to S,\\qquad M(M(a,b),c)=M(a,M(b,c))",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "merge",
          title: "Associativity",
          desc: "임의의 valid states a, b, c에 대해 bracket reassociation이 semantic result를 바꾸지 않아야 한다.",
          metric: "M(M(a,b),c)=M(a,M(b,c))",
        },
        {
          id: "02",
          icon: "shield",
          title: "Closure on State Space",
          desc: "merge 결과는 항상 동일한 state space S 내부에 남아 있어야 한다.",
          metric: "M : S \\times S \\to S",
        },
        {
          id: "03",
          icon: "boxes",
          title: "Valid Partial Representation",
          desc: "각 partial state는 최종 결과를 구성하기 위한 정당한 요약 표현이어야 하며, merge 가능한 의미 보존 state여야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Hierarchical aggregation",
        "Parallel decomposition and merge",
        "Tree-structured reduction",
        "Local summary then global merge",
      ],
    },
    boundary: {
      items: [
        "merge law가 associative하지 않으면 bracket reassociation은 불법이다.",
        "partial summary가 state space에 닫혀 있지 않으면 hierarchical merge를 구성할 수 없다.",
        "state가 순서나 이력에 민감하면 partial merge만으로 원래 의미를 복원하지 못할 수 있다.",
      ],
    },
    relatedConstructions: {
      items: ["WelfordStatsMerge", "OnlineMeanVariance", "TreeReductionState"],
    },
    relatedTransforms: {
      items: [
        "Parallel merge tree",
        "Hierarchical aggregation",
        "Blockwise summary then merge",
      ],
    },
  },
};

export default associativeMerge;