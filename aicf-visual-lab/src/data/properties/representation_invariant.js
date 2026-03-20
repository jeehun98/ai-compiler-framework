// src/data/theory/properties/representation_invariant.js

const representationInvariant = {
  id: "RepresentationInvariant",
  group: "foundational",
  profileKey: "representation_invariant",
  title: "Representation Invariant",
  subtitle: "Representation Invariance Property",
  hero: {
    lead:
      "A computation is representation-invariant when admissible representation changes preserve the same semantic object, allowing equivalent evaluation across multiple valid realizations.",
    canonicalLatex:
      "\\llbracket R(X) \\rrbracket = \\llbracket X \\rrbracket \\Rightarrow F(R(X)) = F(X)",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Semantic Object",
          v: "계산은 특정 물리적 형식이 아니라 논리적 semantic object 또는 index-level value 위에서 정의되어야 한다.",
        },
        {
          k: "Admissible Representation Change",
          v: "표현 변환 R이 semantic object를 보존하면 원래 입력과 동치인 representation으로 간주할 수 있어야 한다.",
        },
        {
          k: "Invariant Evaluation",
          v: "의미를 보존하는 표현 변화 아래에서 계산 결과가 동일해야 한다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "동일한 semantic object를 보존하는 여러 representation 위에서 동일 계산을 정의할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "logical interpretation과 concrete realization을 분리하여 representation-specific lowering을 허용한다.",
        },
      ],
      latex:
        "\\llbracket R(X) \\rrbracket = \\llbracket X \\rrbracket \\Rightarrow F(R(X)) = F(X)",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "boxes",
          title: "Representation Equivalence",
          desc: "표현 변환 전후가 동일한 semantic object를 나타내야 한다.",
          metric: "\\llbracket R(X) \\rrbracket = \\llbracket X \\rrbracket",
        },
        {
          id: "02",
          icon: "shield",
          title: "No Hidden Semantic Reordering",
          desc: "허용된 representation change는 단순 표현 변화여야 하며, 실제 의미 순서를 바꾸는 재배열을 감추면 안 된다.",
        },
        {
          id: "03",
          icon: "target",
          title: "Operator Contract Compatibility",
          desc: "해당 계산이 표현 독립적으로 정의되어 있거나, 허용된 representation class 안에서만 specialization되어야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Representation-specific specialization",
        "View-preserving dispatch",
        "Realization-aware lowering",
        "Redundant representation conversion elimination",
      ],
    },
    boundary: {
      items: [
        "representation 자체가 의미의 일부이면 representation change는 semantic-preserving transform이 아니다.",
        "단순 표현 변화가 아니라 실제 reorder 또는 materialization이 필요하면 이 property만으로는 정당화할 수 없다.",
        "downstream consumer가 특정 concrete contract를 요구하면 representation 자유도는 제한된다.",
      ],
    },
    relatedConstructions: {
      items: ["GEMM", "ElementwiseMap", "ReduceSum", "LayerNorm"],
    },
    relatedTransforms: {
      items: [
        "Representation-aware dispatch",
        "Representation-specialized lowering",
        "Redundant transpose / conversion elimination",
      ],
    },
  },
};

export default representationInvariant;