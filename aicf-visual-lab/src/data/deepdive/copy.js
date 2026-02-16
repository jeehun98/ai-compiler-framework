export const copyDeepDive = {
  id: "Copy",

  kernel_evolution: [
    {
      version: "v1.0",
      tag: "F32 스트리밍 기준",
      throughput: "298.2 GB/s",
      description:
        "순차 Read+Write 패턴의 DRAM 스트리밍 기준점. 이 장비에서 관측된 메모리 대역폭 baseline.",
    },
    {
      version: "v1.0",
      tag: "F16 스칼라 기준",
      throughput: "291.1 GB/s",
      description:
        "2Byte 단위 접근으로 인해 명령 발행 및 트랜잭션 효율 측면에서 미세한 오버헤드 발생. F32 대비 약 2~3% 낮은 대역폭.",
    },
  ],

  profiling_report: {
    F32_기준: "298.2 GB/s (Streaming Baseline)",
    F16_기준: "291.1 GB/s",
    차이: "-2.4%",
    평가_기준: "Copy(F32) 대비 백분율로 Memory Bound 커널 평가",
  },

  analysis:
    "Copy 커널은 이 장비에서 달성 가능한 DRAM 스트리밍 성능의 기준점을 정의한다. Memory Bound 커널(Add, Adam 등)은 이 baseline 대비 90~95% 이상을 목표로 최적화되어야 한다.",
};
