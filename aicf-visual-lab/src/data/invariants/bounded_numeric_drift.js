const boundedNumericDrift = {
  id: "BoundedNumericDrift",
  profileKey: "bounded_numeric_drift",
  group: "numeric",
  title: "제한된 수치 드리프트",
  subtitle: "수치 변화는 통제 가능한 오차 증가 범위 안에서만 허용된다",

  hero: {
    lead:
      "혼합 정밀도, 누적 순서 재배치, online update, 근사적 realization은 사용할 수 있지만, 그로 인해 발생하는 수치 드리프트는 항상 통제 가능하고 안전한 경계 안에 있어야 합니다.",
    canonicalLatex:
      "\\|\\hat{y} - y\\| \\leq \\epsilon",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "통제된 오차 증가",
          v: "실행 방식의 변화로 일부 오차가 생길 수는 있지만, 그 오차는 항상 제한되어 있어야 하며 해석 가능한 범위 안에 있어야 합니다.",
        },
        {
          k: "수치 붕괴 금지",
          v: "overflow, underflow, catastrophic cancellation, 불안정한 증폭은 발생해서는 안 됩니다.",
        },
        {
          k: "보호된 근사 실행",
          v: "근사 realization은 downstream 사용과 호환되는 드리프트 예산 안에서만 허용됩니다.",
        },
      ],
      latex: "\\mathrm{err}(\\hat{y}, y) \\le \\epsilon",
      preview: [
        {
          k: "컴파일러 관점",
          v: "정밀도 완화나 누적 재배치는 제한된 오차 모델 안에서만 유효합니다.",
        },
        {
          k: "커널 관점",
          v: "누산기 정밀도, rescaling 방식, 연산 순서 전략이 드리프트의 안전 여부를 결정합니다.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "gauge",
          title: "범위 안전성",
          desc:
            "실행 전 과정에서 중간값은 항상 안전한 수치 범위 안에 있어야 합니다.",
          metric: "\\max |x| < \\mathrm{safe\\ range}",
        },
        {
          id: "02",
          icon: "scale",
          title: "재배치 드리프트 경계",
          desc:
            "누적 순서나 병합 순서를 바꾸더라도 허용된 오차 예산을 넘어서는 안 됩니다.",
          metric: "\\Delta_{reorder} \\le \\epsilon",
        },
        {
          id: "03",
          icon: "shield",
          title: "안정적 realization 선택",
          desc:
            "runtime은 현재 shape와 scale에 대해 수치적으로 안전한 realization path만 선택해야 합니다.",
          metric: "\\mathrm{safe}(path)=\\text{true}",
        },
      ],
    },

    preserves: {
      items: [
        "제한된 출력 오차",
        "범위 안전한 실행",
        "치명적인 수치 실패 없음",
        "downstream 해석과 양립 가능한 드리프트",
      ],
    },

    failure: {
      items: [
        "변환된 실행 내부에서 overflow 또는 underflow가 발생함",
        "누적 순서 재배치로 인해 허용 불가능한 수준의 드리프트 증가가 발생함",
        "근사 path가 허용된 예산을 넘어 수치적 동작을 바꿔버림",
      ],
    },

    downstreamImpact: {
      items: [
        {
          k: "Logit / Score 안정성",
          v: "과도한 드리프트는 결정 신뢰도와 rank-sensitive 동작을 바꿉니다.",
        },
        {
          k: "Activation Scale",
          v: "통제되지 않은 드리프트는 이후 단계의 norm 기반 또는 threshold 기반 동작을 왜곡합니다.",
        },
      ],
    },

    relatedConstructions: {
      items: [
        { op: "gemm", label: "GEMM 누적" },
        { op: "softmax", label: "Softmax" },
        { op: "layer_norm", label: "LayerNorm 계열 정규화" },
        { op: "adam_step", label: "옵티마이저 업데이트" },
      ],
    },

    relatedTransforms: {
      items: [
        "FP32 누적을 동반한 FP16 경로",
        "재배치된 누적 순서",
        "드리프트 예산이 있는 근사 realization",
        "안정성 guard 기반 path 선택",
      ],
    },
  },
};

export default boundedNumericDrift;