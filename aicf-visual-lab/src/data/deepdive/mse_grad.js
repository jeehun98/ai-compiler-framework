export const mseGradDeepDive = {
  id: "MseGrad",

  kernel_evolution: [
    {
      version: "v1.0",
      tag: "F32 기준 구현",
      throughput: "269.3 GB/s",
      description:
        "2 Read + 1 Write 패턴의 손실 그라디언트 커널. Copy(F32) baseline 대비 약 90% 수준의 유효 대역폭을 달성함.",
    },
    {
      version: "v1.1",
      tag: "F16 벡터화 (vec2)",
      throughput: "267.3 GB/s",
      description:
        "half2 기반 벡터화. Naive 대비 이득은 1~2%로 작으며, 이는 커널이 연산보다 메모리(대역폭/스토어 경로)에 의해 지배됨을 시사함.",
    },
  ],

  profiling_report: {
    F32_유효_대역폭: "269.3 GB/s",
    F16_vec2_유효_대역폭: "267.3 GB/s",
    효율: "약 90% (vs Copy F32 baseline)",
    특이사항: "벡터화 이득 미미 → Pure Memory Bound",
  },

  analysis:
    "MseGrad는 2 Read + 1 Write(3-way traffic) 패턴으로 동작하며, 약 267~269 GB/s는 Copy baseline 대비 90% 수준의 높은 효율이다. 벡터화의 이득이 작다는 점은 이 커널이 연산이 아니라 메모리 대역폭/스토어 경로에 의해 지배되고 있음을 보여준다.",
};
