// src/data/theory/properties/precision_relaxable.js

const precisionRelaxable = {
  id: "PrecisionRelaxable",
  group: "foundational",
  profileKey: "precision_relaxable",
  title: "Precision Relaxable",
  subtitle: "Numerical Relaxation Property",
  hero: {
    lead:
      "A computation is precision-relaxable when reduced numerical precision remains within an admissible semantic error envelope for the intended result.",
    canonicalLatex:
      "\\tilde{F}(X) \\approx F(X),\\qquad d\\big(\\tilde{F}(X),F(X)\\big) \\le \\varepsilon",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Reference Semantics",
          v: "원래 계산 F와 그 결과를 평가할 기준 의미가 정의되어야 한다.",
        },
        {
          k: "Relaxed Evaluation",
          v: "축약된 정밀도의 계산 \\u1d46F 또는 등가 수치 경로가 존재해야 한다.",
        },
        {
          k: "Admissible Error Envelope",
          v: "완화된 계산이 허용 가능한 오차 경계 ε 안에 남아야 한다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "정확한 동치가 아니라 bounded deviation 아래에서 대체 계산을 허용할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "mixed precision, approximate accumulation, precision-specialized lowering을 정당화할 수 있다.",
        },
      ],
      latex:
        "\\tilde{F}(X) \\approx F(X),\\qquad d\\big(\\tilde{F}(X),F(X)\\big) \\le \\varepsilon",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "binary",
          title: "Reference Result",
          desc: "완화 전 기준 계산 F와 비교 의미가 먼저 정의되어야 한다.",
        },
        {
          id: "02",
          icon: "shield",
          title: "Bounded Error",
          desc: "완화된 계산 결과는 허용된 오차 경계 안에 남아 있어야 한다.",
          metric:
            "d\\big(\\tilde{F}(X),F(X)\\big) \\le \\varepsilon",
        },
        {
          id: "03",
          icon: "target",
          title: "Task-Compatible Tolerance",
          desc: "허용 오차는 downstream task 또는 algorithmic contract와 양립 가능해야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Mixed-precision evaluation",
        "Approximate accumulation",
        "Reduced-precision storage or compute",
        "Precision-specialized dispatch",
      ],
    },
    boundary: {
      items: [
        "작은 수치 오차도 의미적 실패로 이어지는 계산에서는 precision relaxation이 허용되지 않을 수 있다.",
        "오차 누적이 허용 envelope를 넘으면 transform은 불법이다.",
        "비교 기준이나 tolerance가 정의되지 않으면 relaxation legality를 주장할 수 없다.",
      ],
    },
    relatedConstructions: {
      items: ["MixedPrecisionGEMM", "FP16Accumulation", "ApproximateNorm"],
    },
    relatedTransforms: {
      items: [
        "Mixed-precision lowering",
        "Reduced-precision accumulation",
        "Tolerance-aware dispatch",
      ],
    },
  },
};

export default precisionRelaxable;