// src/data/theory/weighted_reduction.js
export const weightedReductionTheory = {
  id: "WEIGHTED_REDUCTION",
  title: "Weighted Reduction은 기여도를 따라 정보를 응축한다.",
  subtitle: "Aggregation Geometry & Importance-Weighted Composition",
  hero: {
    lead:
      "Weighted Reduction은 단순한 합산이 아닙니다. 이는 여러 후보의 정보를 동일하게 취급하지 않고, 각 항의 중요도(weight)에 따라 기여를 조절하며 하나의 응축된 표현으로 결합하는 과정입니다. 즉, 어떤 정보가 얼마나 남아야 하는지를 수학적으로 결정하는 가중 집계 연산입니다.",
    canonicalLatex:
      "y = \\sum_{i=1}^{N} w_i v_i, \\qquad \\mathbf{w} \\in \\mathbb{R}^{N}",
  },
  sections: {
    projection: {
      heading: "기여도 기반 정보 응축",
      bullets: [
        {
          k: "Importance Modulation",
          v: "각 항 v_i는 동일한 비중으로 더해지지 않고, weight w_i에 따라 기여도가 조절됩니다.",
        },
        {
          k: "Aggregation into One State",
          v: "다수의 후보 벡터는 weighted sum을 통해 하나의 집계 표현으로 응축됩니다.",
        },
        {
          k: "Context-Dependent Composition",
          v: "출력은 원소 자체의 값뿐 아니라 어떤 원소가 더 중요한지에 대한 weight 구조에 의해 결정됩니다.",
        },
      ],
      latex:
        "V = \\{v_1, \\dots, v_N\\}, \\quad \\mathbf{w} = (w_1, \\dots, w_N) \\quad \\mapsto \\quad y = \\sum_{i=1}^{N} w_i v_i",
      rulesPreview: [
        {
          k: "Convex Aggregation",
          v: "weight가 non-negative이고 합이 1이면 출력은 입력 벡터들의 convex combination이 된다",
        },
        {
          k: "Selective Contribution",
          v: "큰 weight를 가진 항일수록 출력 표현에 더 직접적으로 반영된다",
        },
        {
          k: "Reduction Semantics",
          v: "여러 후보의 정보를 하나의 대표 상태로 응축하는 reduction 구조를 형성한다",
        },
      ],
    },
    equivalence: {
      heading: "Semantic Equivalence Conditions",
      cards: [
        {
          id: "14.1",
          title: "Weight-Mass Consistency",
          desc:
            "weight의 전체 질량 구조가 유지된다면, 집계의 기본적인 기여도 분포는 동일하게 해석될 수 있습니다.",
          metric:
            "\\sum_{i=1}^{N} w_i = \\sum_{i=1}^{N} w'_i",
          note: "Mass Preservation",
          icon: "target",
        },
        {
          id: "14.2",
          title: "Dominant Contribution Preservation",
          desc:
            "가장 큰 기여를 가지는 항들의 순위 구조가 유지된다면, 집계 결과의 의미적 중심은 대체로 보존됩니다.",
          metric:
            "\\mathrm{argsort}(\\mathbf{w}) = \\mathrm{argsort}(\\mathbf{w}')",
          note: "Importance Order Preservation",
          icon: "binary",
        },
        {
          id: "14.3",
          title: "Aggregation Equivalence",
          desc:
            "출력 집계 벡터가 허용 오차 내에서 동일하다면, 개별 weight나 intermediate 값이 달라도 의미적으로 같은 reduction으로 볼 수 있습니다.",
          metric:
            "\\left\\| \\sum_i w_i v_i - \\sum_i w'_i v'_i \\right\\| \\le \\epsilon",
          note: "Output-Level Reduction Equivalence",
          icon: "arrow",
        },
      ],
    },
    cost: {
      heading: "Semantic Distance Measures",
      latex:
        "D_{wr} = w_m \\cdot \\Delta\\mathrm{Mass} + w_o \\cdot \\Delta\\mathrm{Order} + w_a \\cdot \\Delta\\mathrm{Aggregation}",
      pills: [
        {
          title: "Mass Drift",
          tag: "Weight",
          desc:
            "weight 전체 질량 구조가 원래 분포에서 얼마나 벗어나는지",
        },
        {
          title: "Order Drift",
          tag: "Importance",
          desc:
            "주요 기여 항들의 상대적 순위가 얼마나 뒤바뀌는지",
        },
        {
          title: "Aggregation Drift",
          tag: "Output",
          desc:
            "최종 집계 벡터가 원래 reduction 결과에서 얼마나 벗어나는지",
        },
      ],
      foot:
        "Weighted Reduction의 의미 보존은 개별 항의 완전 일치보다, 어떤 정보가 얼마나 기여하는지의 구조와 최종 집계 결과가 얼마나 유지되는지에 달려 있습니다.",
    },
  },
};