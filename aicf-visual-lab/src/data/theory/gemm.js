export const gemmTheory = {
  id: "GEMM",
  title: "GEMM은 정보의 선택과 위상적 압축이다.",
  subtitle: "Mathematical Semantics & Geometry",
  hero: {
    lead:
      "GEMM은 단순한 행렬 곱셈의 합이 아닙니다. 이는 고차원 가설 공간(K)의 정보를 특정 특징 공간(N)으로 투영하며 의미를 확정하는 'Geometric Projection' 단계입니다.",
    canonicalLatex: "C_{ij} = \\langle A_i, B_j \\rangle = \\sum_{k=1}^{K} A_{ik} B_{kj}",
  },
  sections: {
    projection: {
      heading: "가설 공간(K)의 탐색과 소거",
      bullets: [
        { k: "가설의 밀도", v: "차원 K는 샘플 i가 가질 수 있는 잠재적 관계 가설의 개수입니다." },
        { k: "정보의 응축", v: "K개의 독립적 정보를 내적을 통해 하나의 스칼라 결론으로 압축합니다." },
        { k: "의미론적 필터", v: "B의 각 열은 데이터가 만족해야 하는 '의미적 질문'으로 기능합니다." },
      ],
      latex: "A \\in \\mathbb{R}^{M \\times K}, \\quad B \\in \\mathbb{R}^{K \\times N} \\implies C \\in \\mathbb{R}^{M \\times N}",
      rulesPreview: [
        { k: "Basis Compression", v: "의미적 에너지가 집중된 주성분 방향으로 K 차원 축소" },
        { k: "Manifold Projection", v: "데이터의 국소적 위상 구조를 유지하는 저차원 매핑" },
        { k: "Search Reduction", v: "결정적 기여도가 낮은 가설(B_j)의 연산 배제" },
      ],
    },
    equivalence: {
      heading: "위상적 불변성 (Topological Invariance)",
      cards: [
        {
          id: "8.1",
          title: "Rank & Order Invariance",
          desc: "샘플 간의 상대적 거리나 순위가 유지된다면, 두 연산은 기하학적으로 동일한 구조를 가집니다.",
          metric: "\\mathrm{argsort}(C_i) = \\mathrm{argsort}(C'_i)",
          note: "상대적 구조 보존 (Relative Structure Preservation)",
          icon: "target",
        },
        {
          id: "8.2",
          title: "Subspace Equivalence",
          desc: "결과 벡터들이 형성하는 부분공간(Span)이 허용 오차 내에 있다면, 이는 동일한 기저 변환입니다.",
          metric: "\\mathrm{dist}(\\mathcal{S}_{orig}, \\mathcal{S}_{opt}) \\le \\epsilon",
          note: "부분공간 일치성 (Subspace Congruence)",
          icon: "binary",
        },
        {
          id: "8.3",
          title: "Decision Homomorphism",
          desc: "연산 이후의 비선형 임계값을 통과했을 때의 결과(Sign)가 같다면, 내부 오차는 의미를 변화시키지 않습니다.",
          metric: "\\sigma(\\alpha AB) = \\sigma(\\alpha A'B')",
          note: "결정 경계 불변성 (Boundary Invariance)",
          icon: "arrow",
        },
      ],
    },
    cost: {
      heading: "Semantic Entropy Cost",
      latex:
        "Loss_{semantic} = w_r \\cdot \\Delta\\mathrm{Rank} + w_s \\cdot \\Delta\\mathrm{Subspace}",
      pills: [
        { title: "Information Loss", tag: "Entropy", desc: "차원 축소 시 발생하는 정보 엔트로피의 증가량" },
        { title: "Structural Drift", tag: "Topology", desc: "위상 구조가 변형될 확률적 위험도" },
        { title: "Semantic Weight", tag: "Salience", desc: "특정 차원이 전체 추론 결정에 미치는 중요도" },
      ],
      foot:
        "우리는 수치적 정밀도가 아닌, 수학적 본질의 보존 비용을 최소화하는 경로를 탐색합니다.",
    },
  },
};