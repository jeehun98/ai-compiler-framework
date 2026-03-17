// src/data/theory/activation.js
export const activationTheory = {
  id: "ACTIVATION",
  title: "Activation은 표현 공간에 비선형 경계를 부여한다.",
  subtitle: "Decision Geometry & Nonlinear Gating",
  hero: {
    lead:
      "Activation은 단순한 후처리 함수가 아닙니다. 이는 선형 투영된 표현 위에 결정 경계를 도입하여, 어떤 신호는 통과시키고 어떤 신호는 억제함으로써 표현 공간을 비선형적으로 재구성하는 과정입니다.",
    canonicalLatex:
      "y = \\phi(x), \\qquad \\phi \\in \\{\\mathrm{ReLU}, \\mathrm{GELU}, \\mathrm{SiLU}, \\dots\\}",
  },
  sections: {
    projection: {
      heading: "결정 경계와 비선형 공간 분할",
      bullets: [
        {
          k: "Boundary Formation",
          v: "Activation은 입력 공간을 임계값 또는 확률적 전이 구간에 따라 서로 다른 영역으로 분할합니다.",
        },
        {
          k: "Signal Gating",
          v: "일부 신호는 유지하고 일부 신호는 억제하여, 표현의 유효 영역만 다음 계층으로 전달합니다.",
        },
        {
          k: "Nonlinear Expressivity",
          v: "선형 변환만으로는 표현할 수 없는 비선형 결정 구조를 가능하게 합니다.",
        },
      ],
      latex:
        "\\phi : \\mathbb{R}^{N} \\to \\mathbb{R}^{N}, \\qquad y_i = \\phi(x_i)",
      rulesPreview: [
        {
          k: "Half-Space Gating",
          v: "ReLU 계열은 공간을 활성/비활성 반공간으로 나눈다",
        },
        {
          k: "Smooth Transition",
          v: "GELU/SiLU 계열은 경계를 연속적으로 완화해 확률적 게이팅을 만든다",
        },
        {
          k: "Decision Sensitivity",
          v: "작은 입력 차이도 경계 근처에서는 큰 표현 차이로 이어질 수 있다",
        },
      ],
    },
    equivalence: {
      heading: "Semantic Equivalence Conditions",
      cards: [
        {
          id: "13.1",
          title: "Boundary Consistency",
          desc:
            "후행 의사결정에 중요한 경계의 위치가 유지된다면, activation 전후의 의미적 분할은 동일한 구조를 가집니다.",
          metric:
            "\\mathrm{sign}(\\phi(x)) = \\mathrm{sign}(\\phi(x')) \\quad \\text{on decisive regions}",
          note: "Decision Boundary Preservation",
          icon: "target",
        },
        {
          id: "13.2",
          title: "Activation Ordering",
          desc:
            "단조 증가 activation에서는 입력 순서가 유지되어, 후보 간 상대적 크기 구조가 보존됩니다.",
          metric:
            "x_a > x_b \\Rightarrow \\phi(x_a) > \\phi(x_b)",
          note: "Monotone Order Preservation",
          icon: "binary",
        },
        {
          id: "13.3",
          title: "Gating Structure",
          desc:
            "억제되는 영역과 통과되는 영역의 분할 구조가 유지된다면, activation의 본질적 의미는 동일하게 보존됩니다.",
          metric:
            "\\mathbb{1}[\\phi(x) = 0] = \\mathbb{1}[\\phi(x') = 0] \\quad \\text{(ReLU-like)}",
          note: "Active/Inactive Region Consistency",
          icon: "arrow",
        },
      ],
    },
    cost: {
      heading: "Semantic Distance Measures",
      latex:
        "D_{act} = w_b \\cdot \\Delta\\mathrm{Boundary} + w_o \\cdot \\Delta\\mathrm{Order} + w_g \\cdot \\Delta\\mathrm{Gating}",
      pills: [
        {
          title: "Boundary Drift",
          tag: "Decision",
          desc:
            "결정 경계의 위치 또는 경계 근처 반응이 얼마나 달라지는지",
        },
        {
          title: "Order Drift",
          tag: "Monotonicity",
          desc:
            "activation 이후 상대적 순서 구조가 얼마나 변형되는지",
        },
        {
          title: "Gating Drift",
          tag: "Sparsity",
          desc:
            "활성/비활성 영역의 분할이 얼마나 달라지는지",
        },
      ],
      foot:
        "Activation의 의미 보존은 함수값의 완전 일치보다, 경계 구조와 게이팅 패턴이 얼마나 유지되는지에 더 직접적으로 달려 있습니다.",
    },
  },
};