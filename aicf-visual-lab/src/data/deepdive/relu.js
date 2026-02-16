export const reluDeepDive = {
  id: "ReLU",

  kernel_evolution: [
    {
      version: "v0.9",
      tag: "F32 기준 구현",
      throughput: "288.9 GB/s",
      description:
        "ReLU의 기본 구현. Copy(F32) baseline 대비 97% 수준으로 매우 높은 유효 대역폭을 달성함.",
    },
    {
      version: "v1.0",
      tag: "F16 스칼라 (Naive)",
      throughput: "244.8 GB/s",
      description:
        "half 스칼라 처리에서는 벡터화 대비 명령 발행/파이프 효율이 낮아져 메모리 파이프를 충분히 포화시키지 못함(효율 82%).",
    },
    {
      version: "v2.0",
      tag: "F16 벡터화 (half2)",
      throughput: "278.0 GB/s",
      description:
        "half2 벡터화로 로드/스토어 및 연산을 묶어 처리하여 명령 발행 병목을 완화. Copy 대비 93% 수준까지 효율을 회복함.",
    },
  ],

  profiling_report: {
    F32_유효_대역폭: "288.9 GB/s (효율 97%)",
    F16_스칼라_유효_대역폭: "244.8 GB/s (효율 82%)",
    F16_half2_유효_대역폭: "278.0 GB/s (효율 93%)",
    병목_특성: "F16 스칼라에서 Instruction Issue 병목 → half2로 완화",
  },

  analysis:
    "ReLU는 연산 자체가 단순하므로, F16을 스칼라로 처리하면 연산량보다 '명령 발행/파이프 효율'이 먼저 병목이 될 수 있다. half2 벡터화는 로드/스토어와 연산을 묶어 처리해 병목을 완화하며, 결과적으로 메모리 대역폭 효율을 크게 회복한다.",
};
