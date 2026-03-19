const scheduleInvariant = {
  id: "ScheduleInvariant",
  title: "Schedule-Invariant",
  subtitle: "Execution Property",
  hero: {
    lead:
      "A computation is schedule-invariant when different execution schedules preserve the same semantic result.",
    canonicalLatex:
      "Exec_{s_1}(F, X) \\equiv Exec_{s_2}(F, X)",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Meaning",
          v: "실행 순서, 병렬 배치, warp/CTA 역할 배분이 달라도 결과 의미가 동일하다.",
        },
        {
          k: "Use",
          v: "runtime specialization과 schedule selection을 의미 보존과 분리해 다룰 수 있다.",
        },
      ],
      preview: [
        {
          k: "Why It Matters",
          v: "persistent kernel, warp specialization, launch-config dispatch의 기반이 된다.",
        },
        {
          k: "Runtime View",
          v: "입력 shape, hardware 상태에 따라 실행 variant를 바꿀 수 있게 한다.",
        },
      ],
      latex: "Exec_{s_1}(F,X) \\equiv Exec_{s_2}(F,X)",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "workflow",
          title: "Schedule Independence",
          desc: "다른 execution schedule이 동일한 의미 결과를 산출해야 한다.",
        },
        {
          id: "02",
          icon: "shield",
          title: "Race Safety",
          desc: "schedule 변화가 race-sensitive semantic change를 만들지 않아야 한다.",
        },
        {
          id: "03",
          icon: "target",
          title: "Observation Equivalence",
          desc: "외부에서 관찰 가능한 결과 contract가 동일해야 한다.",
        },
      ],
    },
    enables: {
      items: [
        "Persistent kernel selection",
        "Warp specialization",
        "Launch configuration dispatch",
        "Pipeline depth specialization",
      ],
    },
    boundary: {
      items: [
        "순서 민감 accumulation",
        "동기화 누락 시 의미가 깨지는 경우",
        "global mutable state dependency",
      ],
    },
    relatedOps: {
      items: ["GEMM", "ReLU", "AdamStep"],
    },
    relatedTransforms: {
      items: [
        "Persistent execution",
        "Dynamic runtime dispatch",
        "Schedule specialization",
      ],
    },
  },
};

export default scheduleInvariant;