const domainPrunable = {
  id: "DomainPrunable",
  title: "Domain-Prunable",
  subtitle: "Domain Property",
  hero: {
    lead:
      "A computation is domain-prunable when part of the input domain permits semantic simplification or compute elimination.",
    canonicalLatex:
      "x \\in D_0 \\Rightarrow F(x)=c \\;\\text{or}\\; Skip(x)",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "입력 domain 일부에서 계산을 단순화하거나 생략할 수 있다.",
        },
        {
          k: "Typical Cases",
          v: "ReLU, mask, clamp, zero-aware path, sparse-aware execution과 연결된다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "branch pruning, mask-based skip, dead-region elimination의 기반이다.",
        },
        {
          k: "Runtime View",
          v: "실제 입력 분포에 따라 fast path가 열릴 수 있다.",
        },
      ],
      latex: "x\\in D_0 \\Rightarrow F(x)=c\\;\\text{or}\\;Skip(x)",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "target",
          title: "Domain Restriction",
          desc: "특정 입력 영역에서 결과가 고정되거나 단순화 가능해야 한다.",
        },
        {
          id: "02",
          icon: "shield",
          title: "Safe Elimination",
          desc: "생략된 계산이 최종 의미를 바꾸지 않아야 한다.",
        },
        {
          id: "03",
          icon: "binary",
          title: "Guard Validity",
          desc: "어떤 입력이 pruneable domain에 속하는지 판단 기준이 유효해야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Branch elimination",
        "Mask-aware skip",
        "Zero-region pruning",
        "Sparse-aware execution",
      ],
    },
    boundary: {
      items: [
        "global nonlinear coupling",
        "skip가 downstream semantics를 바꾸는 경우",
        "guard cost가 prune benefit보다 큰 경우",
      ],
    },
    relatedOps: {
      items: ["ReLU", "Softmax"],
    },
    relatedTransforms: {
      items: [
        "Dead-region pruning",
        "Mask specialization",
        "Sparse-aware execution",
      ],
    },
  },
};

export default domainPrunable;