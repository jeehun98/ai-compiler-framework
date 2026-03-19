const orderRewritable = {
  id: "OrderRewritable",
  title: "Order-Rewritable",
  subtitle: "Reduction Property",
  hero: {
    lead:
      "A computation is order-rewritable when its execution order can be rearranged without changing the preserved semantic result.",
    canonicalLatex: "F(x_1, x_2, \\dots, x_n) = F(\\pi(x_1, x_2, \\dots, x_n))",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "연산의 계산 순서를 바꾸어도 보존되는 의미가 동일하다.",
        },
        {
          k: "Typical Basis",
          v: "대개 associative 성질에 기대며, 경우에 따라 commutative 성질도 함께 요구된다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "reduction reorder, split accumulation, tree reduction의 정당화 근거가 된다.",
        },
        {
          k: "Compiler View",
          v: "semantic legality와 execution order를 분리하여 다룰 수 있게 한다.",
        },
      ],
      latex: "F(X) = F(\\pi(X))",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "arrow",
          title: "Permutation Safety",
          desc: "입력 순서 재배치가 최종 의미를 바꾸지 않아야 한다.",
          metric: "F(X) = F(\\pi(X))",
        },
        {
          id: "02",
          icon: "merge",
          title: "Merge Compatibility",
          desc: "순서를 바꾼 부분 결과들이 다시 유효하게 합쳐질 수 있어야 한다.",
        },
        {
          id: "03",
          icon: "shield",
          title: "Semantic Stability",
          desc: "단순한 실행 순서 변경이 아니라 의미 보존적 재구성임이 보장되어야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Tree reduction",
        "Split-K style accumulation",
        "Parallel reduction merge",
        "Reduction reordering",
      ],
    },
    boundary: {
      items: [
        "prefix dependency가 있는 경우",
        "순서 자체가 의미인 recurrence",
        "strict sequential state update",
      ],
    },
    relatedOps: {
      items: ["LayerNorm", "Softmax", "GEMM"],
    },
    relatedTransforms: {
      items: [
        "Reduction reordering",
        "Associative parallelization",
        "Blockwise partial merge",
      ],
    },
  },
};

export default orderRewritable;