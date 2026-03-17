// src/data/theory/softmax.js
export const softmaxTheory = {
  id: "SOFTMAX",
  title: "Softmax는 경쟁 에너지를 확률 분포로 정규화한다.",
  subtitle: "Probability Geometry & Competitive Normalization",
  hero: {
    lead:
      "Softmax는 단순한 지수 함수의 조합이 아닙니다. 이는 각 후보의 상대적 에너지를 동일한 simplex 위의 확률 분포로 사상하여, 경쟁 관계를 정규화된 선택 구조로 바꾸는 과정입니다.",
    canonicalLatex:
      "p_{i,j} = \\frac{e^{x_{i,j}}}{\\sum_{k=1}^{N} e^{x_{i,k}}}",
  },
  sections: {
    projection: {
      heading: "경쟁 에너지의 확률 공간 사상",
      bullets: [
        {
          k: "Relative Energy",
          v: "Softmax는 절대값 자체보다 같은 row 안의 상대적 에너지 차이를 확률 구조로 변환합니다.",
        },
        {
          k: "Simplex Projection",
          v: "각 row의 출력은 합이 1인 확률 분포로 정규화되어 probability simplex 위에 놓입니다.",
        },
        {
          k: "Competition Structure",
          v: "한 후보의 확률이 커질수록 다른 후보의 질량은 줄어들어, 경쟁적 선택 구조가 형성됩니다.",
        },
      ],
      latex:
        "x_i \\in \\mathbb{R}^{N} \\quad \\mapsto \\quad p_i \\in \\Delta^{N-1}, \\qquad \\Delta^{N-1} = \\left\\{ p \\in \\mathbb{R}^{N} \\mid p_j \\ge 0, \\sum_j p_j = 1 \\right\\}",
      rulesPreview: [
        {
          k: "Translation Invariance",
          v: "같은 row에 상수를 더하거나 빼도 확률 구조는 변하지 않는다",
        },
        {
          k: "Order Preservation",
          v: "더 큰 logit은 더 큰 probability로 대응된다",
        },
        {
          k: "Mass Conservation",
          v: "전체 확률 질량은 항상 1로 유지된다",
        },
      ],
    },
    equivalence: {
      heading: "Semantic Equivalence Conditions",
      cards: [
        {
          id: "10.1",
          title: "Translation Invariance",
          desc: "같은 row의 모든 원소에 동일한 상수를 더하거나 빼더라도 softmax 결과는 변하지 않습니다. 이는 수치 안정화의 핵심 근거가 됩니다.",
          metric:
            "\\mathrm{Softmax}(x_i) = \\mathrm{Softmax}(x_i - c_i \\mathbf{1})",
          note: "Shift-Equivalent Probability Mapping",
          icon: "target",
        },
        {
          id: "10.2",
          title: "Simplex Constraint",
          desc: "출력은 항상 non-negative이며, 전체 합이 1인 확률 분포를 형성합니다. 이는 softmax가 probability geometry 위의 연산임을 의미합니다.",
          metric:
            "\\sum_{j=1}^{N} p_{i,j} = 1, \\qquad p_{i,j} \\ge 0",
          note: "Probability Mass Conservation",
          icon: "binary",
        },
        {
          id: "10.3",
          title: "Order Monotonicity",
          desc: "동일 row 내에서 더 큰 logit은 더 큰 probability로 대응됩니다. 따라서 softmax는 후보 간의 순위 구조를 보존합니다.",
          metric:
            "x_{i,a} > x_{i,b} \\Rightarrow p_{i,a} > p_{i,b}",
          note: "Competitive Order Preservation",
          icon: "arrow",
        },
      ],
    },
    cost: {
      heading: "Semantic Distance Measures",
      latex:
        "D_{softmax} = w_s \\cdot \\Delta\\mathrm{Simplex} + w_o \\cdot \\Delta\\mathrm{Order} + w_d \\cdot \\Delta\\mathrm{Distribution}",
      pills: [
        {
          title: "Simplex Drift",
          tag: "Mass",
          desc: "확률 합이 1에서 벗어나는 정도",
        },
        {
          title: "Order Drift",
          tag: "Rank",
          desc: "후보 간 상대적 순위가 뒤바뀌는 정도",
        },
        {
          title: "Distribution Drift",
          tag: "KL",
          desc: "원래 분포와 근사 분포 사이의 거리",
        },
      ],
      foot:
        "Softmax의 본질은 개별 수치의 완전 일치보다, 확률 질량 보존과 후보 간 경쟁 구조가 유지되는지에 있습니다.",
    },
  },
};