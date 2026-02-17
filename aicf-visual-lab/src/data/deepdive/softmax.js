export const softmaxDeepDive = {
  id: "SOFTMAX",

  // KernelDeepDive.jsx에서 data.kernel_evolution으로 접근
  kernel_evolution: [
    {
      version: "v1.0",
      tag: "Naive Global Reduce",
      throughput: "Low",
      description:
        "전체 행렬에 대해 Global Memory Atomic 연산을 사용하여 Max와 Sum을 계산. 동기화 오버헤드가 크고 캐시 효율이 떨어짐.",
    },
    {
      version: "v2.0",
      tag: "One Block Per Row",
      throughput: "~180.0 GB/s",
      description:
        "가장 널리 쓰이는 최적화 패턴. 각 CUDA 블록이 하나의 행(Row)을 전담하여, Shared Memory 내에서 Max/Sum 리덕션을 수행함. 글로벌 메모리 왕복을 최소화.",
    },
    {
      version: "v3.0",
      tag: "Warp Shuffle & Vectorization",
      throughput: "Target: >250 GB/s",
      description:
        "현재 구현은 float 단위로 읽고 있음. 향후 float4/half2 벡터 로드와 Warp Shuffle 최적화를 적용하여 레지스터 압박을 줄이고 메모리 대역폭을 한계치까지 끌어올릴 예정.",
    },
  ],

  // KernelDeepDive.jsx에서 data.profiling_report로 접근
  profiling_report: {
    DRAM_대역폭_활용: "51.6%", // RTX 3060 기준 (185/360)
    SM_점유율: "High",
    Shared_Memory: "Used (Reduction)",
    수치_안정성: "Safe (Sub-Max)",
    FP16_가속: "Mixed Precision",
  },

  // KernelDeepDive.jsx에서 data.analysis로 접근
  analysis:
    "Softmax는 전형적인 '3-Pass' 알고리즘(Max 탐색 -> Sum 계산 -> Normalize)을 가진다. v2 구현은 이를 하나의 커널로 융합(Fusion)하여 데이터를 레지스터와 공유 메모리에 유지함으로써 성능을 확보했다. 특히 FP16 입력에 대해 내부 연산은 FP32로 처리하여 Overflow를 방지하고 정밀도를 유지한다.",
};