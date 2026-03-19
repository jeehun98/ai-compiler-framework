const rematerializable = {
  id: "Rematerializable",
  title: "Rematerializable",
  subtitle: "Memory–Compute Bridge",
  hero: {
    lead:
      "A computation is rematerializable when an intermediate can be recomputed later instead of being stored explicitly.",
    canonicalLatex:
      "Store(z) \\;\\leftrightarrow\\; Recompute(z=g(x))",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "중간값을 저장하지 않고 나중에 다시 계산할 수 있다.",
        },
        {
          k: "Trade-off",
          v: "memory를 줄이는 대신 compute를 늘리는 교환 구조를 가진다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "checkpointing, saved tensor minimization, epilogue intermediate elimination을 가능하게 한다.",
        },
        {
          k: "System View",
          v: "memory optimization과 compute overhead를 직접 연결하는 property다.",
        },
      ],
      latex: "Store(z)\\;\\leftrightarrow\\;Recompute(z=g(x))",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "binary",
          title: "Recompute Availability",
          desc: "중간값이 upstream 정보로부터 다시 생성 가능해야 한다.",
        },
        {
          id: "02",
          icon: "shield",
          title: "Observation Safety",
          desc: "저장 생략이 externally observable semantic을 깨면 안 된다.",
        },
        {
          id: "03",
          icon: "merge",
          title: "State Independence",
          desc: "재계산이 숨은 mutable state나 비결정적 상태에 의존하면 안 된다.",
        },
      ],
    },
    enables: {
      items: [
        "Activation checkpointing",
        "Backward recompute",
        "Saved tensor minimization",
        "Intermediate elimination",
      ],
    },
    boundary: {
      items: [
        "stateful / random dependency",
        "재계산 비용이 지나치게 큰 경우",
        "intermediate가 외부 contract인 경우",
      ],
    },
    relatedOps: {
      items: ["LayerNorm", "AdamStep", "GEMM"],
    },
    relatedTransforms: {
      items: [
        "Checkpointing",
        "Recompute-heavy low-memory mode",
        "Intermediate elimination",
      ],
    },
  },
};

export default rematerializable;