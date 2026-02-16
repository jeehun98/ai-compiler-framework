export const biasAddDeepDive = {
  id: "Add",

  kernel_evolution: [
    {
      version: "v1.0",
      tag: "F32 기본 구현 (Baseline)",
      throughput: "314.7 GB/s",
      description:
        "4Byte 단위 접근으로 메모리 트랜잭션 효율이 높음. 이론적 메모리 대역폭에 근접하는 기준 성능을 형성함.",
    },
    {
      version: "v2.0",
      tag: "F16 스칼라 (Naive)",
      throughput: "293.6 GB/s",
      description:
        "데이터 총 이동량이 절반으로 감소하여 실행 시간은 약 2배 단축되었으나, 잦은 메모리 요청으로 인해 대역폭 효율은 소폭 하락함.",
    },
    {
      version: "v3.0",
      tag: "F16 벡터화 (half2)",
      throughput: "310.1 GB/s",
      description:
        "half2 인트린직을 통해 명령어 발행 비용을 절감. F32 수준의 대역폭 효율을 회복하며 DRAM 활용률 93.8%에 도달함.",
    },
  ],

  profiling_report: {
    DRAM_대역폭_활용률: "93.8%",
    SM_파이프라인_활용률: "18.7%",
    L2_캐시_적중률: "91.5%",
    메모리_액세스_효율: "98.9%",
  },

  analysis:
    "Memory Bound 커널의 핵심은 연산 최적화가 아니라 대역폭 포화이다. F16 도입으로 데이터 이동량을 절반으로 줄여 실행 시간을 크게 단축했고, half2 벡터화로 인스트럭션 병목을 완화하여 DRAM 활용률을 극대화했다.",
};
