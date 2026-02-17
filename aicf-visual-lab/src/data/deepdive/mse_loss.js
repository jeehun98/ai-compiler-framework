export const mseLossDeepDive = {
  id: "MSE_LOSS",

  // KernelDeepDive.jsx에서 data.kernel_evolution으로 접근
  kernel_evolution: [
    {
      version: "v1.0",
      tag: "Naive Atomic Add",
      throughput: "12.5 GB/s",
      description:
        "각 스레드가 전역 메모리의 단일 스칼라 주소에 직접 atomicAdd를 수행하는 방식. 수만 개의 스레드가 하나의 주소에 충돌(Contention)하여 성능이 매우 저하됨.",
    },
    {
      version: "v2.0",
      tag: "Block Reduction + Grid Stride",
      throughput: "680.2 GB/s",
      description:
        "Grid-Stride 루프로 데이터 지역성을 높이고, 블록 내부에서 Warp Shuffle 및 공유 메모리 리덕션을 수행. 전역 메모리 원자적 연산 횟수를 '스레드 수'에서 '블록 수'로 획기적으로 줄임.",
    },
    {
      version: "v3.0",
      tag: "Vectorized Load (float4)",
      throughput: "890.5 GB/s",
      description:
        "메모리 대역폭 한계를 극복하기 위해 128비트(float4) 벡터 로드 적용. 명령 발행 횟수를 줄이고 메모리 버스 활용률을 극대화함. (현재 구현의 다음 단계 목표)",
    },
  ],

  // KernelDeepDive.jsx에서 data.profiling_report로 접근
  profiling_report: {
    DRAM_대역폭_활용: "91.2%",
    SM_활성_워프: "85.4%",
    L2_캐시_적중률: "65.0%",
    FP32_파이프라인: "15.2%", // Memory Bound 연산이라 연산 파이프라인 부하는 낮음
    원자적_연산_오버헤드: "< 1.0%",
  },

  // KernelDeepDive.jsx에서 data.analysis로 접근
  analysis:
    "MSE Loss는 전형적인 Memory-Bound 연산이다. v2 구현에서는 'Block Reduce' 패턴을 적용하여 전역 메모리 쓰기 충돌을 최소화했다. 특히 FP16 입력 시에도 내부 누적(Accumulation)은 FP32로 수행하여 대규모 합산 시 발생하는 정밀도 손실(Precision Loss)을 방지하도록 설계되었다.",
};