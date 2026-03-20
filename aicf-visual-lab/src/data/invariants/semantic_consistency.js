const semanticConsistency = {
  id: "SemanticConsistency",
  profileKey: "semantic_consistency",
  group: "semantic",
  title: "Semantic Consistency",
  subtitle: "Meaning Preservation Invariant",

  hero: {
    lead:
      "허용된 변환과 서로 다른 runtime realization 이후에도, 연산이 외부에 대해 가지는 의미적 결과는 동일하게 유지되어야 합니다.",
    canonicalLatex: "f'(x) \\equiv f(x)",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Observable Equivalence",
          v: "변환된 실행 경로는 외부에서 관측되는 계산 의미를 동일하게 유지해야 합니다.",
        },
        {
          k: "Not Just Shape Equality",
          v: "shape, dtype, memory layout이 비슷하다고 해서 semantic equivalence가 보장되지는 않습니다.",
        },
        {
          k: "Task-Level Meaning",
          v: "downstream에서 기대하는 contract가 유지될 때만 transform은 진정한 의미 보존으로 간주됩니다.",
        },
      ],
      latex: "Interp(f'(x)) = Interp(f(x))",
      preview: [
        {
          k: "Compiler View",
          v: "semantic consistency는 transform legality의 최종 기준입니다.",
        },
        {
          k: "Runtime View",
          v: "경로는 달라질 수 있어도 결과 해석은 동일해야 합니다.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "shield",
          title: "Output Meaning Equivalence",
          desc:
            "출력 tensor가 downstream에서 동일한 의미로 해석될 수 있어야 합니다.",
          metric: "Meaning(y') = Meaning(y)",
          note: "No semantic drift",
        },
        {
          id: "02",
          icon: "target",
          title: "Dependency-Consistent Result",
          desc:
            "중간 realization이 달라도 dependency가 표현하는 실제 계산 의미는 보존되어야 합니다.",
          metric: "Dep'(x) \\sim Dep(x)",
        },
        {
          id: "03",
          icon: "lock",
          title: "Contract Preservation",
          desc:
            "operator-level contract 또는 subgraph-level contract가 깨지면 transform은 허용되지 않습니다.",
          metric: "Contract(f') = Contract(f)",
        },
      ],
    },

    preserves: {
      items: [
        "Observable output meaning",
        "Downstream-valid interpretation",
        "Task-level contract consistency",
        "Equivalent dependency meaning",
      ],
    },

    failure: {
      items: [
        "shape는 유지되지만 function meaning이 달라지는 rewrite",
        "approximation이 rank / threshold / selection behavior를 바꾸는 경우",
        "numerically close해 보여도 downstream semantic contract가 깨지는 경우",
      ],
    },

    relatedConstructions: {
      items: [
        { op: "gemm", label: "GEMM" },
        { op: "softmax", label: "Softmax" },
        { op: "layernorm", label: "LayerNorm" },
        { op: "relu", label: "ReLU" },
      ],
    },

    relatedTransforms: {
      items: [
        "Legal fusion only under semantic equivalence guard",
        "Alternative realization allowed only if the same external meaning is preserved",
        "Runtime specialization must remain contract-safe",
      ],
    },
  },
};

export default semanticConsistency;