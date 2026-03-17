export const gemmTheory = {
  id: "GEMM",
  title: "GEMM은 정보의 선택과 표현 공간 투영이다.",
  subtitle: "Mathematical Semantics & Geometry",
  hero: {
    lead:
      "GEMM은 단순한 수치 연산의 반복이 아니라, 입력 표현을 잠재 가설 공간(K)을 거쳐 출력 특징 공간(N)으로 사상하는 선형 투영입니다. 이 과정에서 다수의 기여는 reduction을 통해 응축되며, 결과는 이후 연산이 해석할 새로운 표현 공간을 형성합니다.",
    canonicalLatex:
      "C_{ij} = \\langle A_i, B_j \\rangle = \\sum_{k=1}^{K} A_{ik} B_{kj}",
  },
  sections: {
    projection: {
      heading: "가설 공간(K)의 투영과 응축",
      bullets: [
        {
          k: "가설의 밀도",
          v: "차원 K는 샘플 i가 가질 수 있는 잠재적 관계 가설의 수를 나타냅니다.",
        },
        {
          k: "정보의 응축",
          v: "K개의 기여는 reduction을 통해 하나의 출력 스칼라로 응축됩니다.",
        },
        {
          k: "의미론적 필터",
          v: "B의 각 열은 입력이 어떤 출력 특징으로 사상될지를 결정하는 선형 질문으로 기능합니다.",
        },
      ],
      latex:
        "A \\in \\mathbb{R}^{M \\times K}, \\quad B \\in \\mathbb{R}^{K \\times N} \\implies C \\in \\mathbb{R}^{M \\times N}",
      rulesPreview: [
        {
          k: "Projection View",
          v: "입력 표현을 출력 특징 부분공간으로 사상하는 선형 투영",
        },
        {
          k: "Bilinear Interaction",
          v: "A의 행과 B의 열 사이의 관계를 내적으로 측정",
        },
        {
          k: "Reduction Semantics",
          v: "K축의 다중 기여를 하나의 출력 스칼라로 응축",
        },
      ],
    },
    equivalence: {
      heading: "Semantic Equivalence Conditions",
      cards: [
        {
          id: "8.1",
          title: "Rank & Order Invariance",
          desc: "출력의 상대적 순서 구조가 유지된다면, 두 투영은 동일한 선택 구조를 보존합니다.",
          metric: "\\mathrm{argsort}(C_i) = \\mathrm{argsort}(C'_i)",
          note: "상대적 구조 보존 (Relative Structure Preservation)",
          icon: "target",
        },
        {
          id: "8.2",
          title: "Subspace Equivalence",
          desc: "결과 벡터들이 형성하는 부분공간이 허용 오차 내에 있다면, 두 연산은 동일한 표현 공간을 형성합니다.",
          metric: "\\mathrm{dist}(\\mathcal{S}_{orig}, \\mathcal{S}_{opt}) \\le \\epsilon",
          note: "부분공간 일치성 (Subspace Congruence)",
          icon: "binary",
        },
        {
          id: "8.3",
          title: "Decision Homomorphism",
          desc: "후행 비선형 경계에서의 의사결정 결과가 동일하다면, 내부 수치 오차는 의미를 바꾸지 않습니다.",
          metric: "\\sigma(\\alpha AB) = \\sigma(\\alpha A'B')",
          note: "결정 경계 불변성 (Boundary Invariance)",
          icon: "arrow",
        },
      ],
    },
    cost: {
      heading: "Semantic Distance Measures",
      latex:
        "D_{semantic} = w_r \\cdot \\Delta\\mathrm{Rank} + w_s \\cdot \\Delta\\mathrm{Subspace} + w_b \\cdot \\Delta\\mathrm{Boundary}",
      pills: [
        {
          title: "Order Drift",
          tag: "Rank",
          desc: "출력 순위 구조가 변형되는 정도",
        },
        {
          title: "Subspace Drift",
          tag: "Geometry",
          desc: "표현이 형성하는 부분공간이 벗어나는 정도",
        },
        {
          title: "Boundary Drift",
          tag: "Decision",
          desc: "후행 의사결정 경계에서 결과가 달라질 위험도",
        },
      ],
      foot:
        "우리는 수치값의 완전 일치보다, 표현 구조와 결정 경계가 얼마나 유지되는지를 더 본질적인 기준으로 다룹니다.",
    },
  },
};