// src/data/theory/state_merge.js
export const stateMergeTheory = {
  id: "STATE_MERGE",
  title: "상태 병합: 기억과 교정의 가법 합성",
  subtitle: "Pure Mathematics & Geometric Coupling",
  hero: {
    lead:
      "가산(Addition)은 단순한 산술적 합산이 아닙니다. 이는 기준 상태(Memory)와 보정 상태(Correction)가 같은 좌표계에서 결합되어 새로운 표현 상태를 형성하는 과정입니다. 특히 residual 구조에서는 기존 표현을 유지한 채 새로운 변화량만을 누적하는 수학적 병합으로 해석할 수 있습니다.",
    canonicalLatex:
      "Y = R + X, \\quad R, X \\in \\mathbb{R}^{M \\times N}",
  },
  sections: {
    projection: {
      heading: "가법 합성과 상태 결합",
      bullets: [
        {
          k: "State Composition",
          v: "보정 신호(X)는 기존 상태(R)에 누적되어 새로운 표현 상태를 형성합니다.",
        },
        {
          k: "Identity Retention",
          v: "기준 상태(R)는 완전히 대체되지 않고 병합 결과 안에 직접 보존됩니다.",
        },
        {
          k: "Coordinate Alignment",
          v: "두 상태는 동일한 좌표계와 shape 위에서 결합되어야 의미 보존적 병합이 성립합니다.",
        },
      ],
      latex:
        "Y_{ij} = R_{ij} + X_{ij} \\quad \\implies \\quad \\text{Merged State in a Shared Coordinate Frame}",
      rulesPreview: [
        {
          k: "Additive Composition",
          v: "새로운 상태는 base state와 corrective state의 가법 합성으로 정의된다",
        },
        {
          k: "Identity Path",
          v: "기준 상태는 병합 이후에도 직접적인 기여 항으로 남는다",
        },
        {
          k: "Merge Legality",
          v: "동일 좌표계와 정렬된 표현 사이에서만 의미 보존적 결합이 가능하다",
        },
      ],
    },
    equivalence: {
      heading: "Semantic Equivalence Conditions",
      cards: [
        {
          id: "9.1",
          title: "Additive Composition",
          desc: "두 상태가 같은 좌표계에서 elementwise하게 합성된다면, 병합은 수학적으로 정당한 상태 결합이다.",
          metric: "Y_{ij} = R_{ij} + X_{ij}",
          note: "Elementwise Merge Legality",
          icon: "target",
        },
        {
          id: "9.2",
          title: "Identity Preservation",
          desc: "기준 상태의 기여가 직접적으로 남아 있다면, 병합 이후에도 원래 경로의 의미는 보존된다.",
          metric: "\\frac{\\partial Y}{\\partial R} = 1",
          note: "Residual Path Integrity",
          icon: "arrow",
        },
        {
          id: "9.3",
          title: "Coordinate Alignment",
          desc: "두 상태가 동일한 shape과 표현 프레임을 공유할 때만 의미 보존적 병합이 성립한다.",
          metric: "\\mathrm{shape}(R) = \\mathrm{shape}(X)",
          note: "Frame Consistency",
          icon: "binary",
        },
      ],
    },
  },
};