const layoutFlexible = {
  id: "LayoutFlexible",
  title: "Layout-Flexible",
  subtitle: "Representation Property",
  hero: {
    lead:
      "A computation is layout-flexible when representation changes do not alter the preserved semantic meaning of the computation.",
    canonicalLatex:
      "F(X) \\equiv F(\\mathrm{view}(X))",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "메모리 배치나 표현 방식이 달라도 계산 의미가 동일하다.",
        },
        {
          k: "Consequence",
          v: "layout specialization, transpose elimination, vectorization이 가능해진다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "contiguous fast path, packed load/store, tensor-core mapping의 출발점이다.",
        },
        {
          k: "Compiler View",
          v: "semantic tensor view와 physical storage view를 분리할 수 있게 한다.",
        },
      ],
      latex: "F(X) \\equiv F(\\mathrm{view}(X))",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "boxes",
          title: "Representation Equivalence",
          desc: "표현 변화가 같은 tensor meaning을 유지해야 한다.",
        },
        {
          id: "02",
          icon: "shield",
          title: "View Safety",
          desc: "view reinterpretation이 실제 semantic reorder를 숨기면 안 된다.",
        },
        {
          id: "03",
          icon: "target",
          title: "Access Compatibility",
          desc: "layout specialization이 연산 contract와 충돌하지 않아야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Layout specialization",
        "Vectorized path selection",
        "Tensor-core friendly mapping",
        "View / transpose elimination",
      ],
    },
    boundary: {
      items: [
        "layout 자체가 semantic meaning에 포함되는 경우",
        "실제 reorder / copy가 필요한 경우",
        "consumer가 특정 layout contract를 요구하는 경우",
      ],
    },
    relatedOps: {
      items: ["GEMM", "AdamStep"],
    },
    relatedTransforms: {
      items: [
        "Contiguous fast path",
        "Vectorization",
        "Representation-aware dispatch",
      ],
    },
  },
};

export default layoutFlexible;