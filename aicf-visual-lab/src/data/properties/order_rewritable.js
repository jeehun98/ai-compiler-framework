// src/data/theory/properties/order_rewritable.js

const orderRewritable = {
  id: "OrderRewritable",
  group: "foundational",
  profileKey: "order_rewritable",
  title: "Order Rewritable",
  subtitle: "Ordering Invariance Property",
  hero: {
    lead:
      "A computation is order-rewritable when evaluation order may be rearranged without changing the preserved semantic result, provided the governing dependency law is respected.",
    canonicalLatex:
      "\\pi \\in \\Pi_{valid} \\Rightarrow F(x_1,\\dots,x_n)=F(x_{\\pi(1)},\\dots,x_{\\pi(n)})",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Admissible Reordering",
          v: "연산 순서를 바꿀 수 있는 유효한 순열 집합 Π_valid 가 정의되어야 한다.",
        },
        {
          k: "Semantic Invariance",
          v: "허용된 순서 변화 아래에서 계산 결과는 동일해야 한다.",
        },
        {
          k: "Dependency Preservation",
          v: "재배치는 데이터 또는 의미 의존성을 깨지 않는 범위에서만 허용된다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "계산을 여러 순서로 재표현해도 동일한 semantic object를 유지할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "loop interchange, reassociation, traversal reordering, evaluation schedule rewrite를 정당화할 수 있다.",
        },
      ],
      latex:
        "\\pi \\in \\Pi_{valid} \\Rightarrow F(x_1,\\dots,x_n)=F(x_{\\pi(1)},\\dots,x_{\\pi(n)})",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "arrow",
          title: "Admissible Permutation",
          desc: "재배치는 허용된 dependency-preserving 순열에 속해야 한다.",
          metric: "\\pi \\in \\Pi_{valid}",
        },
        {
          id: "02",
          icon: "shield",
          title: "Semantic Invariance",
          desc: "허용된 순서 변화는 최종 semantic result를 바꾸지 않아야 한다.",
        },
        {
          id: "03",
          icon: "target",
          title: "Dependency Respect",
          desc: "필수적인 causality, dataflow, reduction law를 깨는 재배치는 불법이다.",
        },
      ],
    },
    enables: {
      items: [
        "Loop reordering",
        "Traversal rewrite",
        "Reassociation under dependency law",
        "Order-specialized evaluation",
      ],
    },
    boundary: {
      items: [
        "계산 결과가 순서 자체에 의존하면 reorder는 semantic-preserving transform이 아니다.",
        "side effect 또는 history-sensitive state가 존재하면 order rewrite의 자유도는 제한된다.",
        "허용되지 않은 permutation이 hidden dependency를 깨면 transform은 불법이다.",
      ],
    },
    relatedConstructions: {
      items: ["ReduceSum", "CommutativeAggregation", "OrderIndependentMapReduce"],
    },
    relatedTransforms: {
      items: [
        "Loop interchange",
        "Dependency-preserving reorder",
        "Associative / commutative reassociation",
      ],
    },
  },
};

export default orderRewritable;