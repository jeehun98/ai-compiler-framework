export const biasTheory = {
  id: "BiasAdd",
  title: "덧셈은 연산이 아니라, 상태의 병합과 교정이다.",
  subtitle: "State Merge & Manifold Coupling",
  hero: {
    lead: "AICF는 모든 가산(Addition)을 단순 원소별 계산이 아닌, 두 표현 상태(State)의 결합으로 정의합니다. Bias는 에너지의 오프셋을, Residual은 기억과 오차의 보정을 의미합니다.",
    canonicalLatex: "Y = X + R \\quad (\\text{where } R \\text{ is Identity/Memory})",
  },
  sections: {
    projection: {
      heading: "평행 이동과 매니폴드 결합",
      bullets: [
        { k: "Translation", v: "BiasAdd는 표현 공간의 위상 구조를 유지한 채 원점만을 이동시키는 평행 이동입니다." },
        { k: "Coupling", v: "ResidualAdd는 이전 계층의 '기억(R)'과 현재의 '교정(X)'을 결합하여 매니폴드를 재구성합니다." },
        { k: "Stability", v: "덧셈은 정보의 발산을 억제하고 표현 공간을 안정화하는 장치로 기능합니다." },
      ],
      latex: "\\text{SRR} = \\mathbb{E}\\left[\\frac{\|X\|}{\|R\|}\\right] \\implies \\text{if SRR} \\ll 1, \\text{ then } Y \\approx R",
      rulesPreview: [
        { k: "Precision Asymmetry", v: "기억(R)은 고정밀 유지, 미세 교정(X)은 낮은 정밀도 허용" },
        { k: "Dead-zone Truncation", v: "통계적 확신 하에 연산 블록 전체를 상수로 대체" },
        { k: "Alignment Guard", v: "분포 불일치 감지 시 SafeAdd 또는 FusedNorm 전환" },
      ],
    },
    equivalence: {
      heading: "상태 보존과 결정 경계 (Boundary Stability)",
      cards: [
        {
          id: "4.1",
          title: "Topological Invariance",
          desc: "BiasAdd는 공간 내 모든 점 사이의 거리와 kNN 구조를 완벽하게 보존합니다.",
          metric: "\\text{dist}(x_i, x_j) = \\text{dist}(x_i+b, x_j+b)",
          note: "Pairwise Distance 보존",
          icon: "target",
        },
        {
          id: "4.2",
          title: "Decision Stability",
          desc: "ResidualAdd 이후에도 데이터의 상대적 순위가 유지된다면 결정 경계는 안정적입니다.",
          metric: "\\text{sign}(X+R) = \\text{sign}(R) \\text{ (if } X \\text{ is small)}",
          note: "결정 경계 불변성",
          icon: "arrow",
        },
        {
          id: "5.2",
          title: "Origin Rigidity",
          desc: "후속 연산이 절대 위치에 민감하지 않다면(Shift-invariant), 적극적인 Folding을 허용합니다.",
          metric: "\\text{rigid}(Y) \\in \\{low, high\\}",
          note: "원점 강성 모델",
          icon: "binary",
        },
      ],
    },
    cost: {
      heading: "Semantic Stability Cost",
      latex: "Cost_{merge} = \\omega_s \\cdot \\Delta\\text{SRR} + \\omega_a \\cdot \\text{AlignmentViolation}",
      pills: [
        { title: "Merge Stability", tag: "SRR", desc: "기억과 교정이 결합될 때 발생하는 신호 왜곡 위험" },
        { title: "Alignment Cost", tag: "Variance", desc: "두 공간의 분포 불일치를 교정하기 위한 자원" },
        { title: "Gradient Flow", tag: "Jacobian", desc: "역전파 시 정보 소실 없이 기울기를 전달하는 비용" },
      ],
      foot: "AICF는 단순 대역폭이 아닌, '상태 병합의 안정성'을 최우선 비용 지표로 삼습니다.",
    },
  },
};