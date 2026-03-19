const associativeMerge = {
  id: "AssociativeMerge",
  title: "Associative-Merge",
  subtitle: "Merge Property",
  hero: {
    lead:
      "A computation is associative-merge when partial results can be grouped and merged in different bracketings while preserving the same semantic result.",
    canonicalLatex:
      "M(M(a,b),c) = M(a,M(b,c))",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "부분 결과를 어떤 묶음 구조로 합쳐도 동일한 결과를 얻을 수 있다.",
        },
        {
          k: "Core Role",
          v: "병렬 계산 후 partial state를 안전하게 merge할 수 있게 해준다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "blockwise reduction, hierarchical merge, multi-stage accumulation의 핵심 전제다.",
        },
        {
          k: "Compiler View",
          v: "global reduction을 local partial state + merge tree로 분해할 수 있게 한다.",
        },
      ],
      latex: "M(M(a,b),c) = M(a,M(b,c))",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "merge",
          title: "Bracket Invariance",
          desc: "묶음 구조가 달라도 merge 결과가 같아야 한다.",
          metric: "M(M(a,b),c)=M(a,M(b,c))",
        },
        {
          id: "02",
          icon: "boxes",
          title: "Partial Validity",
          desc: "각 partial result가 최종 결과의 유효한 중간 표현이어야 한다.",
        },
        {
          id: "03",
          icon: "shield",
          title: "Merge Closure",
          desc: "merge 결과가 다시 같은 merge space에 남아 있어야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Hierarchical reduction",
        "Parallel partial merge",
        "Blockwise state accumulation",
        "Welford-style merge pipelines",
      ],
    },
    boundary: {
      items: [
        "merge operator가 associative하지 않은 경우",
        "partial result format이 merge closure를 가지지 않는 경우",
        "history-sensitive state aggregation",
      ],
    },
    relatedOps: {
      items: ["LayerNorm", "StateMerge", "WeightedReduction"],
    },
    relatedTransforms: {
      items: [
        "Parallel merge tree",
        "Hierarchical aggregation",
        "Blockwise reduction merge",
      ],
    },
  },
};

export default associativeMerge;