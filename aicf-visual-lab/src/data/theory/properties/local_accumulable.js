const localAccumulable = {
  id: "LocalAccumulable",
  title: "Local-Accumulable",
  subtitle: "Accumulation Property",
  hero: {
    lead:
      "A computation is local-accumulable when intermediate contributions can be accumulated locally before a later global materialization.",
    canonicalLatex:
      "y = Writeback\\left(\\sum_t a_t\\right)",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "중간 기여값을 register/shared memory 같은 로컬 공간에 먼저 누적할 수 있다.",
        },
        {
          k: "Benefit",
          v: "global write 횟수를 줄이고 locality를 높인다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "fused local accumulation, delayed writeback, blockwise reduction에 중요하다.",
        },
        {
          k: "System View",
          v: "실행 구조를 global-materialize-first에서 local-accumulate-first로 바꿀 수 있게 한다.",
        },
      ],
      latex: "y = Writeback\\left(\\sum_t a_t\\right)",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "merge",
          title: "Accumulate Safety",
          desc: "로컬 누적 결과가 최종 semantic result와 일치해야 한다.",
        },
        {
          id: "02",
          icon: "shield",
          title: "Delayed Writeback Safety",
          desc: "writeback 지연이 의미 contract를 바꾸지 않아야 한다.",
        },
        {
          id: "03",
          icon: "boxes",
          title: "Local Capacity Fit",
          desc: "로컬 누적이 가능한 표현과 범위를 가져야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Register accumulation",
        "Shared-memory accumulation",
        "Delayed global writeback",
        "Fused block compute",
      ],
    },
    boundary: {
      items: [
        "중간 상태가 외부적으로 관찰되어야 하는 경우",
        "즉시 materialization이 필요한 경우",
        "local storage footprint가 과도한 경우",
      ],
    },
    relatedOps: {
      items: ["GEMM", "Softmax"],
    },
    relatedTransforms: {
      items: [
        "Local accumulation",
        "Writeback minimization",
        "Blockwise fused compute",
      ],
    },
  },
};

export default localAccumulable;