// src/data/theory/layer_norm.js
export const layerNormTheory = {
  id: "LAYER_NORM",
  title: "LayerNorm은 표현의 절대 에너지를 제거하고 상대 구조를 남긴다.",
  subtitle: "Normalization Geometry & Sample-Local Reparameterization",
  hero: {
    lead:
      "LayerNorm은 단순한 평균과 분산 계산이 아닙니다. 이는 각 샘플 내부의 feature 분포를 재중심화하고 재스케일링하여, 표현의 절대 크기보다 상대적 구조가 더 직접적으로 드러나도록 만드는 재매개변수화 과정입니다.",
    canonicalLatex:
      "y_{i,j} = \\gamma_j \\cdot \\frac{x_{i,j} - \\mu_i}{\\sqrt{\\sigma_i^2 + \\epsilon}} + \\beta_j",
  },
  sections: {
    projection: {
      heading: "샘플 내부 분포의 재중심화와 재스케일링",
      bullets: [
        {
          k: "Sample-Local Statistics",
          v: "LayerNorm은 각 샘플 i 내부의 feature 축만을 사용해 평균과 분산을 계산하며, 다른 샘플의 상태에는 의존하지 않습니다.",
        },
        {
          k: "Energy Removal",
          v: "입력의 절대적 offset과 global scale을 줄여 표현의 전체 에너지보다 내부 상대 구조가 더 직접적으로 드러나게 만듭니다.",
        },
        {
          k: "Affine Reparameterization",
          v: "정규화 이후의 표현은 gamma와 beta를 통해 다시 학습 가능한 좌표계로 재매개변수화됩니다.",
        },
      ],
      latex:
        "\\mu_i = \\frac{1}{N}\\sum_{j=1}^{N} x_{i,j}, \\qquad \\sigma_i^2 = \\frac{1}{N}\\sum_{j=1}^{N}(x_{i,j}-\\mu_i)^2",
      rulesPreview: [
        {
          k: "Centering",
          v: "샘플 내부 평균을 제거해 표현을 원점 기준으로 다시 정렬한다",
        },
        {
          k: "Scaling",
          v: "feature 축 분산을 정규화해 과도한 에너지 편차를 줄인다",
        },
        {
          k: "Reparameterization",
          v: "정규화된 표현을 affine 파라미터를 통해 다시 학습 가능한 형태로 복원한다",
        },
      ],
    },
    equivalence: {
      heading: "Semantic Equivalence Conditions",
      cards: [
        {
          id: "11.1",
          title: "Sample-Locality",
          desc: "각 샘플의 정규화 결과는 오직 그 샘플 내부 feature들에 의해서만 결정됩니다. 따라서 LayerNorm은 batch 구성 변화와 무관한 local normalization입니다.",
          metric:
            "\\mu_i, \\sigma_i^2 \\text{ depend only on } \\{x_{i,1}, \\dots, x_{i,N}\\}",
          note: "Per-Sample Statistic Independence",
          icon: "target",
        },
        {
          id: "11.2",
          title: "Centering Consistency",
          desc: "정규화된 표현은 샘플 내부 평균이 0이 되도록 재중심화됩니다. 이는 절대 offset보다 상대적 deviation을 보존하기 위한 핵심 조건입니다.",
          metric:
            "\\frac{1}{N}\\sum_{j=1}^{N}\\frac{x_{i,j}-\\mu_i}{\\sqrt{\\sigma_i^2+\\epsilon}} \\approx 0",
          note: "Zero-Mean Recentered Representation",
          icon: "binary",
        },
        {
          id: "11.3",
          title: "Scale-Normalized Structure",
          desc: "정규화 이후 각 샘플의 표현은 분산 기준으로 재조정되어, 과도한 scale 차이가 후행 연산의 민감도로 직접 전달되는 것을 줄입니다.",
          metric:
            "\\frac{1}{N}\\sum_{j=1}^{N}\\left(\\frac{x_{i,j}-\\mu_i}{\\sqrt{\\sigma_i^2+\\epsilon}}\\right)^2 \\approx 1",
          note: "Unit-Scale Internal Geometry",
          icon: "arrow",
        },
      ],
    },
    cost: {
      heading: "Semantic Distance Measures",
      latex:
        "D_{ln} = w_m \\cdot \\Delta\\mathrm{Mean} + w_v \\cdot \\Delta\\mathrm{Variance} + w_a \\cdot \\Delta\\mathrm{Affine}",
      pills: [
        {
          title: "Mean Drift",
          tag: "Center",
          desc: "정규화 후 샘플 평균이 0에서 벗어나는 정도",
        },
        {
          title: "Variance Drift",
          tag: "Scale",
          desc: "정규화 후 샘플 분산이 기준 scale에서 벗어나는 정도",
        },
        {
          title: "Affine Drift",
          tag: "Restore",
          desc: "gamma, beta 적용 이후 재매개변수화 구조가 흔들리는 정도",
        },
      ],
      foot:
        "LayerNorm의 본질은 입력값 자체의 완전 일치보다, 샘플 내부 분포가 얼마나 안정적으로 재중심화되고 재스케일링되는지에 있습니다.",
    },
  },
};