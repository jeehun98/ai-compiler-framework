// src/data/theory/properties/schedule_invariant.js

const scheduleInvariant = {
  id: "ScheduleInvariant",
  group: "foundational",
  profileKey: "schedule_invariant",
  title: "Schedule Invariant",
  subtitle: "Execution Independence Property",
  hero: {
    lead:
      "A computation is schedule-invariant when differences in execution schedule do not alter the preserved semantic result, provided all required dependencies are respected.",
    canonicalLatex:
      "\\sigma_1,\\sigma_2 \\in \\Sigma_{valid} \\Rightarrow \\llbracket F \\rrbracket_{\\sigma_1}=\\llbracket F \\rrbracket_{\\sigma_2}",
  },
  sections: {
    definition: {
      bullets: [
        {
          k: "Valid Schedule Set",
          v: "의존성을 만족하는 유효한 실행 스케줄 집합 Σ_valid 가 정의되어야 한다.",
        },
        {
          k: "Execution Independence",
          v: "유효한 스케줄 차이는 계산의 semantic result를 바꾸지 않아야 한다.",
        },
        {
          k: "Dependency-Constrained Freedom",
          v: "자유로운 스케줄링은 필수 의존성 보존 아래에서만 허용된다.",
        },
      ],
      preview: [
        {
          k: "Mathematical Consequence",
          v: "같은 dependency graph를 갖는 서로 다른 실행 순서들이 동일한 의미를 생성할 수 있다.",
        },
        {
          k: "Compilation Consequence",
          v: "parallel scheduling, async overlap, pipeline staging, work partitioning을 정당화할 수 있다.",
        },
      ],
      latex:
        "\\sigma_1,\\sigma_2 \\in \\Sigma_{valid} \\Rightarrow \\llbracket F \\rrbracket_{\\sigma_1}=\\llbracket F \\rrbracket_{\\sigma_2}",
    },
    legality: {
      cards: [
        {
          id: "01",
          icon: "workflow",
          title: "Valid Schedule Membership",
          desc: "스케줄은 dependency-preserving execution set 안에 속해야 한다.",
          metric: "\\sigma \\in \\Sigma_{valid}",
        },
        {
          id: "02",
          icon: "shield",
          title: "Semantic Invariance",
          desc: "유효한 스케줄 차이는 계산의 의미를 바꾸지 않아야 한다.",
        },
        {
          id: "03",
          icon: "target",
          title: "No Schedule-Visible Side Effect",
          desc: "실행 타이밍이나 interleaving 자체가 observable semantics가 되면 schedule invariance는 깨진다.",
        },
      ],
    },
    enables: {
      items: [
        "Parallel work scheduling",
        "Pipeline staging",
        "Async overlap",
        "Dependency-preserving work redistribution",
      ],
    },
    boundary: {
      items: [
        "observable side effect가 execution order에 노출되면 schedule invariance는 성립하지 않는다.",
        "race-sensitive state가 존재하면 서로 다른 interleaving은 동일 의미를 보장하지 못한다.",
        "required dependency를 위반하는 스케줄은 valid schedule set에 속하지 않는다.",
      ],
    },
    relatedConstructions: {
      items: ["PipelineExecution", "AsyncReduction", "DependencyGraphScheduler"],
    },
    relatedTransforms: {
      items: [
        "Parallel scheduling",
        "Async staging",
        "Dependency-aware pipeline overlap",
      ],
    },
  },
};

export default scheduleInvariant;