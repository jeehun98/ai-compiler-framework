// src/data/theory/attention.js
export const attentionTheory = {
  id: "ATTENTION",
  title: "Attention은 관계 에너지로 정보를 재배치한다.",
  subtitle: "Relational Projection & Weighted Information Routing",
  hero: {
    lead:
      "Attention은 단순한 가중 평균이 아닙니다. 이는 query와 key 사이의 관계 에너지를 계산하고, 그 경쟁 구조를 확률 분포로 정규화한 뒤, value 공간의 정보를 선택적으로 재배치하는 관계 기반 투영 과정입니다.",
    canonicalLatex:
      "\\mathrm{Attention}(Q,K,V) = \\mathrm{Softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right)V",
  },
  sections: {
    projection: {
      heading: "관계 에너지 기반 정보 라우팅",
      bullets: [
        {
          k: "Relational Energy",
          v: "Query와 Key의 내적은 두 토큰 사이의 관계 에너지를 측정합니다.",
        },
        {
          k: "Competitive Normalization",
          v: "Softmax는 각 query에 대해 key 후보들의 경쟁 구조를 확률 분포로 변환합니다.",
        },
        {
          k: "Information Routing",
          v: "확률 가중치는 value 공간의 정보를 선택적으로 집계하여 새로운 표현을 형성합니다.",
        },
      ],
      latex:
        "A = \\mathrm{Softmax}\\left(\\frac{QK^T}{\\sqrt{d_k}}\\right), \\qquad Y = AV",
      rulesPreview: [
        {
          k: "Relational Projection",
          v: "query는 key 공간 위에서 관계 기반 투영을 수행한다",
        },
        {
          k: "Weighted Aggregation",
          v: "value 정보는 attention 확률에 따라 가중 평균으로 결합된다",
        },
        {
          k: "Context Formation",
          v: "출력 표현은 주변 토큰 정보가 반영된 새로운 context 벡터가 된다",
        },
      ],
    },

    equivalence: {
      heading: "Semantic Equivalence Conditions",
      cards: [
        {
          id: "12.1",
          title: "Translation Invariance (Logit Shift)",
          desc:
            "attention logit에 동일한 상수를 더하거나 빼더라도 softmax 확률 구조는 변하지 않습니다.",
          metric:
            "\\mathrm{Softmax}(Z_i) = \\mathrm{Softmax}(Z_i - c_i)",
          note: "Attention Logit Shift Invariance",
          icon: "target",
        },
        {
          id: "12.2",
          title: "Probability Mass Conservation",
          desc:
            "각 query에 대한 attention weight는 항상 합이 1인 확률 분포를 형성합니다.",
          metric:
            "\\sum_j A_{i,j} = 1",
          note: "Attention Simplex Constraint",
          icon: "binary",
        },
        {
          id: "12.3",
          title: "Weighted Value Consistency",
          desc:
            "출력은 value 벡터들의 가중 평균으로 구성되며, attention weight에 의해 정보 흐름이 결정됩니다.",
          metric:
            "Y_i = \\sum_j A_{i,j} V_j",
          note: "Value Aggregation Semantics",
          icon: "arrow",
        },
      ],
    },

    cost: {
      heading: "Semantic Distance Measures",
      latex:
        "D_{attn} = w_p \\cdot \\Delta\\mathrm{Prob} + w_o \\cdot \\Delta\\mathrm{Order} + w_v \\cdot \\Delta\\mathrm{Value}",
      pills: [
        {
          title: "Probability Drift",
          tag: "Distribution",
          desc:
            "attention 확률 분포가 원래 경쟁 구조에서 얼마나 벗어나는지",
        },
        {
          title: "Order Drift",
          tag: "Rank",
          desc:
            "attention weight의 순위 구조가 변형되는 정도",
        },
        {
          title: "Value Aggregation Drift",
          tag: "Context",
          desc:
            "value 정보의 가중 결합 결과가 얼마나 변형되는지",
        },
      ],
      foot:
        "Attention의 의미 보존은 개별 수치의 완전 일치보다, 관계 확률 구조와 정보 집계 결과가 얼마나 유지되는지에 의해 판단됩니다.",
    },
  },
};