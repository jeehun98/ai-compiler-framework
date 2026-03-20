// src/data/theory/properties/domain_prunable.js

const domainPrunable = {
  id: "DomainPrunable",
  group: "foundational",
  profileKey: "domain_prunable",
  title: "Domain Prunable",
  subtitle: "Domain Restriction Property",
  hero: {
    lead:
      "A computation is domain-prunable when there exists a restricted input region on which the original function can be replaced by a semantically equivalent simpler form.",
    canonicalLatex:
      "\\exists D_0 \\subseteq D,\\; \\exists G\\; \\text{s.t.}\\; x \\in D_0 \\Rightarrow F(x)=G(x)",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Restricted Domain",
          v: "전체 입력 공간 D 안에 의미적으로 단순화 가능한 부분집합 D₀가 존재해야 한다.",
        },
        {
          k: "Equivalent Reduced Form",
          v: "x ∈ D₀ 에 대해 원래 계산 F(x)는 더 단순한 표현 G(x)와 의미적으로 동치여야 한다.",
        },
        {
          k: "Guardable Membership",
          v: "입력이 D₀에 속하는지 판단하는 조건이 정의 가능해야 한다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "입력 domain의 일부에서는 원래 함수 대신 더 단순한 동치 표현을 사용할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "guarded specialization, branch pruning, region-specific lowering을 구성할 수 있다.",
        },
      ],
      latex:
        "\\exists D_0 \\subseteq D,\\; \\exists G\\; \\text{s.t.}\\; x \\in D_0 \\Rightarrow F(x)=G(x)",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "target",
          title: "Restricted Semantic Region",
          desc: "입력의 특정 부분집합 D₀ 위에서 함수가 더 단순한 의미 형태로 축약 가능해야 한다.",
          metric: "x \\in D_0 \\Rightarrow F(x)=G(x)",
        },
        {
          id: "02",
          icon: "shield",
          title: "Semantic Equivalence",
          desc: "축약된 형태 G는 해당 영역에서 원래 계산 F와 동일한 결과를 만들어야 한다.",
          metric: "\\forall x \\in D_0,\\; F(x)=G(x)",
        },
        {
          id: "03",
          icon: "binary",
          title: "Guard Validity",
          desc: "입력이 단순화 가능한 영역에 속하는지 판단하는 guard가 유효하고 의미적으로 정당해야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Guarded specialization",
        "Branch pruning",
        "Region-specific simplification",
        "Sparse / zero-aware execution",
      ],
    },
    boundary: {
      items: [
        "출력 의미가 전체 domain에 걸쳐 강하게 결합되어 있으면 지역적 simplification이 성립하지 않을 수 있다.",
        "축약된 branch가 downstream semantics를 바꾸면 pruning은 불법이다.",
        "domain membership guard가 원래 의미와 정확히 대응하지 않으면 specialization은 불법이다.",
      ],
    },
    relatedConstructions: {
      items: ["ReLU", "Clamp", "MaskedSelect", "ZeroAwareMultiply"],
    },
    relatedTransforms: {
      items: [
        "Dead-region pruning",
        "Mask-guided specialization",
        "Sparse-aware lowering",
      ],
    },
  },
};

export default domainPrunable;