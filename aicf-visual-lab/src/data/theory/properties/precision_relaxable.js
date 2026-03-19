const precisionRelaxable = {
  id: "PrecisionRelaxable",
  title: "Precision-Relaxable",
  subtitle: "Numerical Property",
  hero: {
    lead:
      "A computation is precision-relaxable when lower precision or bounded approximation is acceptable under a preserved semantic contract.",
    canonicalLatex:
      "\\|F(x)-\\tilde{F}(x)\\| \\leq \\epsilon",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "낮은 정밀도나 bounded approximation이 허용 범위 안에서 받아들여질 수 있다.",
        },
        {
          k: "Important Distinction",
          v: "exact-preserving legality와는 별도의 numerical contract가 필요하다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "mixed precision, fast math, reduced accumulation path를 가능하게 한다.",
        },
        {
          k: "Risk",
          v: "semantic legality와 numerical error budget을 분리하지 않으면 검증 구조가 무너진다.",
        },
      ],
      latex: "\\|F(x)-\\tilde{F}(x)\\| \\le \\epsilon",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "target",
          title: "Bounded Error",
          desc: "오차가 정의된 허용 범위 안에 있어야 한다.",
          metric: "\\|F(x)-\\tilde{F}(x)\\|\\le\\epsilon",
        },
        {
          id: "02",
          icon: "shield",
          title: "Contract Awareness",
          desc: "precision relaxation이 downstream contract를 깨지 않아야 한다.",
        },
        {
          id: "03",
          icon: "binary",
          title: "Stability Guard",
          desc: "불안정한 수치 영역을 별도로 방어할 수 있어야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Mixed precision execution",
        "Approximate intrinsic path",
        "Reduced-precision accumulation",
        "Fast-math dispatch",
      ],
    },
    boundary: {
      items: [
        "strict exact semantics required",
        "error amplification이 큰 recurrence",
        "stability-critical region",
      ],
    },
    relatedOps: {
      items: ["Softmax", "GEMM", "LayerNorm"],
    },
    relatedTransforms: {
      items: [
        "FP16/BF16 specialization",
        "Fast math dispatch",
        "Bounded numerical transform",
      ],
    },
  },
};

export default precisionRelaxable;