// src/data/theory/state_merge.js
export const stateMergeTheory = {
  id: "STATE_MERGE",
  title: "상태 병합: 기억과 교정의 수치적 결합",
  subtitle: "Pure Mathematics & Geometric Coupling",
  hero: {
    lead:
      "가산(Addition)은 단순한 산술적 합산이 아닙니다. 이는 서로 다른 기하학적 성질을 가진 두 상태(State)가 만나 새로운 평행을 이루는 과정이며, 기준 신호(Memory)를 보정 신호(Correction)로 수정하여 정보의 유효성을 확정하는 단계입니다.",
    canonicalLatex: "Y = X + R \\quad \\text{s.t. } R \\in \\text{Manifold}_{stable}",
  },
  sections: {
    projection: {
      heading: "평행 이동과 매니폴드 결합",
      bullets: [
        {
          k: "Information Supplement",
          v: "교정 신호(X)는 기존의 안정적 표현(R)이 설명하지 못한 잔차 오차를 보완합니다.",
        },
        {
          k: "Translation Principle",
          v: "상수 신호의 가산은 표현 공간의 위상 구조를 유지하며 원점만을 이동시키는 에너지 오프셋입니다.",
        },
        {
          k: "Manifold Coupling",
          v: "두 표현 공간이 결합될 때 정보의 유효 부피(Volume)가 재구성되며 분포의 정렬이 발생합니다.",
        },
      ],
      latex:
        "\\text{SRR} = \\mathbb{E}\\left[ \\frac{\\|X\\|}{\\|R\\|} \\right] \\implies \\text{Semantic Sensitivity}",
      rulesPreview: [
        { k: "Precision Asymmetry", v: "기억(R)의 고정밀 상태 보존과 미세 교정(X)의 저정밀 완화" },
        { k: "Semantic Truncation", v: "에너지 합이 논리적 임계값 미만일 때 수치적 Zero로 확정" },
        { k: "Alignment Constraint", v: "분포 불일치(Misalignment) 감지 시 정밀도 상향 및 결합 전략 변경" },
      ],
    },
    equivalence: {
      heading: "수학적 불변성 (Mathematical Invariance)",
      cards: [
        {
          id: "Rule 1",
          title: "Topological Translation",
          desc: "공간 내 모든 점 사이의 상대적 거리와 근접 구조(kNN)를 완벽하게 보존하는 평행 이동.",
          metric: "\\| (x_i + b) - (x_j + b) \\| = \\| x_i - x_j \\|",
          note: "Pairwise Structure Invariance",
          icon: "target",
        },
        {
          id: "Rule 2",
          title: "Boundary Stability",
          desc: "신호 결합 후에도 데이터의 논리적 결정 경계(Sign)가 유지되는 범위 내에서의 수치적 동일성.",
          metric: "\\mathrm{sign}(X+R) = \\mathrm{sign}(R)",
          note: "Decision Boundary Stability",
          icon: "arrow",
        },
        {
          id: "Rule 3",
          title: "Origin Rigidity",
          desc: "후속 연산의 절대 위치 민감도에 따른 원점 좌표계의 고정 강도 정의.",
          metric: "\\text{Rigidity} \\in \\{low, high\\}",
          note: "Shift-invariance Analysis",
          icon: "binary",
        },
      ],
    },
  },
};