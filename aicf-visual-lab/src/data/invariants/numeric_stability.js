const numericStability = {
  id: "NumericStability",
  profileKey: "numeric_stability",
  group: "numeric",
  title: "Numeric Stability",
  subtitle: "Stable Numeric Behavior Invariant",

  hero: {
    lead:
      "실행 형태가 달라지더라도 overflow, underflow, catastrophic cancellation 없이 수치적으로 안정된 결과를 유지할 수 있어야 합니다.",
    canonicalLatex:
      "\\lvert \\hat{y} - y \\rvert \\leq \\epsilon \\quad \\text{under stable realization}",
  },

  sections: {
    meaning: {
      bullets: [
        {
          k: "Bounded Deviation",
          v: "실행 경로 변화가 허용되더라도 오차는 안정적으로 제어 가능해야 합니다.",
        },
        {
          k: "Stable Reduction",
          v: "accumulation, normalization, weighted sum은 수치적으로 무너지지 않는 형태여야 합니다.",
        },
        {
          k: "Scale-Aware Realization",
          v: "입력 scale이나 dynamic range 변화에 대해 비정상적인 발산이 없어야 합니다.",
        },
      ],
      latex: "\\mathrm{err}(\\hat{y}, y) \\le \\epsilon",
      preview: [
        {
          k: "Compiler View",
          v: "precision relaxation이나 reduction reordering은 stability guard 아래에서만 허용됩니다.",
        },
        {
          k: "Runtime View",
          v: "shape와 device 조건에 맞춰 더 안정적인 path를 선택해야 할 수 있습니다.",
        },
      ],
    },

    guard: {
      cards: [
        {
          id: "01",
          icon: "gauge",
          title: "Overflow / Underflow Resistance",
          desc:
            "실행 과정에서 수치 범위를 벗어나지 않는 realization이어야 합니다.",
          metric: "\\max |x| < \\mathrm{safe\\ range}",
          note: "Range-aware path",
        },
        {
          id: "02",
          icon: "scale",
          title: "Stable Accumulation Order",
          desc:
            "reduction이나 merge 순서가 바뀌더라도 허용 가능한 안정성 범위를 유지해야 합니다.",
          metric: "\\Delta_{reorder} \\le \\epsilon",
        },
        {
          id: "03",
          icon: "shield",
          title: "Normalization Safety",
          desc:
            "softmax, norm, probability-like structure에서 안정적인 normalization이 유지되어야 합니다.",
          metric: "\\sum_i p_i = 1",
        },
      ],
    },

    preserves: {
      items: [
        "Bounded numeric error",
        "Stable accumulation behavior",
        "Normalization safety",
        "Range-aware execution correctness",
      ],
    },

    failure: {
      items: [
        "reduction reorder 이후 catastrophic cancellation이 커지는 경우",
        "softmax / norm에서 overflow 또는 underflow가 발생하는 경우",
        "precision relaxation으로 인해 허용 범위를 넘어서는 numeric drift가 발생하는 경우",
      ],
    },

    relatedConstructions: {
      items: [
        { op: "softmax", label: "Softmax" },
        { op: "layernorm", label: "LayerNorm" },
        { op: "rmsnorm", label: "RMSNorm" },
        { op: "gemm", label: "GEMM Accumulation" },
      ],
    },

    relatedTransforms: {
      items: [
        "Online normalization with stable rescaling",
        "Higher-precision accumulation for reduced drift",
        "Stability-guarded precision relaxation",
      ],
    },
  },
};

export default numericStability;