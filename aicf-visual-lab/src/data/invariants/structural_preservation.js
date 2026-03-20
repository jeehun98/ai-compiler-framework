const structuralPreservation = {
  id: "StructuralPreservation",
  profileKey: "structural_preservation",
  group: "structural",
  title: "Structural Preservation",
  subtitle: "Dependency and Shape Relation Invariant",

  hero: {
    lead:
      "변환 이후에도 연산이 전제하는 shape relation, dependency structure, reduction contract가 깨지지 않아야 합니다.",
    canonicalLatex: "\\mathcal{S}(f'(x)) \\cong \\mathcal{S}(f(x))",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Shape Relation Preservation",
          v: "단순 shape equality뿐 아니라 차원 간 관계와 해석 가능한 구조가 유지되어야 합니다.",
        },
        {
          k: "Dependency Preservation",
          v: "어떤 값이 어떤 입력과 reduction axis에 의존하는지 구조적으로 깨지면 안 됩니다.",
        },
        {
          k: "Reduction Contract",
          v: "local accumulation, merge, tile decomposition 이후에도 원래 reduction 의미가 유지되어야 합니다.",
        },
      ],
      latex: "Dep'(y) \\cong Dep(y)",
      preview: [
        {
          k: "Compiler View",
          v: "tiling, decomposition, fusion은 구조적 contract를 보존할 때만 허용됩니다.",
        },
        {
          k: "Runtime View",
          v: "layout이나 path는 달라져도 dependency graph의 의미는 유지되어야 합니다.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "binary",
          title: "Shape Contract",
          desc:
            "출력 및 중간 구조가 기대된 차원 관계를 유지해야 합니다.",
          metric: "\\mathrm{ShapeRel}(y') = \\mathrm{ShapeRel}(y)",
          note: "Shape relation must survive",
        },
        {
          id: "02",
          icon: "orbit",
          title: "Dependency Graph Preservation",
          desc:
            "어떤 출력이 어떤 입력 집합과 연결되는지 dependency structure가 깨지면 안 됩니다.",
          metric: "G_{dep}' \\cong G_{dep}",
        },
        {
          id: "03",
          icon: "zap",
          title: "Reduction Integrity",
          desc:
            "분해된 partial computation을 다시 합쳤을 때 원래 reduction 의미가 유지되어야 합니다.",
          metric: "\\bigoplus_i p_i = R(x)",
        },
      ],
    },

    preserves: {
      items: [
        "Shape relation semantics",
        "Dependency graph integrity",
        "Reduction contract preservation",
        "Composable structural meaning",
      ],
    },

    failure: {
      items: [
        "tiling 이후 axis meaning이 깨지는 경우",
        "layout transform이 downstream interpretation을 훼손하는 경우",
        "partial accumulation merge가 원래 reduction contract를 만족하지 못하는 경우",
      ],
    },

    relatedConstructions: {
      items: [
        { op: "gemm", label: "GEMM" },
        { op: "matmul", label: "MatMul" },
        { op: "softmax", label: "Softmax" },
        { op: "attention", label: "Attention" },
      ],
    },

    relatedTransforms: {
      items: [
        "Tile decomposition with dependency-safe merge",
        "Layout rewrite under preserved axis semantics",
        "Local accumulation followed by structure-preserving merge",
      ],
    },
  },
};

export default structuralPreservation;