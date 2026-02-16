export const batchnormDeepDive = {
  id: "BatchNorm",

  kernel_evolution: [
    {
      version: "v0.1 (Baseline)",
      tag: "Naive Atomic (Inference)",
      throughput: "183.2 GB/s",
      description:
        "기능 구현 중심의 초기 버전. NCHW 레이아웃에서 기본적인 병렬 처리를 수행하나 벡터화 부재 및 디버그 로그 오버헤드로 인해 대역폭 효율이 낮음 (약 60%).",
    },
    {
      version: "v0.1 (Baseline)",
      tag: "Naive Atomic (Training)",
      throughput: "2.4 GB/s ❌",
      description:
        "Global Atomic 연산을 직접 사용. NCHW 특성상 수천 개의 스레드가 동일 주소에 충돌(Contention)하여 심각한 성능 저하 발생. Block Reduction 도입이 시급함.",
    },
  ],

  profiling_report: {
    유효_메모리_대역폭_Infer: "183.2 GB/s",
    유효_메모리_대역폭_Train: "2.4 GB/s (Bottleneck)",
    주요_병목: "Global Atomic Contention",
    최적화_필요: "Warp/Block Reduction",
  },

  analysis:
    "현재 구현은 NCHW 레이아웃에서 Global Atomic을 사용할 때 발생하는 '직렬화(Serialization)' 문제를 적나라하게 보여준다. Training 성능이 2.4GB/s로 곤두박질친 것은 수천 개의 스레드가 하나의 주소에 락을 거는 현상 때문이며, 이를 해결하기 위해 계층적 리덕션(Hierarchical Reduction) 구조로의 변경이 필수적이다.",
};