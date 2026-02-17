export const crossEntropyDeepDive = {
  id: "CROSS_ENTROPY",

  // KernelDeepDive.jsx에서 data.kernel_evolution으로 접근
  kernel_evolution: [
    {
      version: "v1.0",
      tag: "Naive Composition",
      throughput: "35.2 GB/s",
      description:
        "Softmax, Log, NLLLoss를 별도의 커널로 순차 실행. 중간 결과(Probability Matrix)를 메모리에 쓰고 읽는 비용으로 인해 대역폭 낭비가 심함.",
    },
    {
      version: "v2.0",
      tag: "Fused Kernel (One Block per Row)",
      throughput: "152.4 GB/s",
      description:
        "LogSumExp와 NLLLoss를 하나의 커널로 융합. 샘플(N) 당 하나의 블록을 할당하여 레지스터 내에서 Max/Sum을 계산하고, Global Memory 접근을 최소화함.",
    },
    {
      version: "v3.0",
      tag: "Online Softmax & Vectorization",
      throughput: "On Roadmap",
      description:
        "데이터 로드 시 float4 벡터화 적용 및 Warp Shuffle을 최적화하여 레지스터 압박을 줄이고 점유율(Occupancy)을 높일 계획.",
    },
  ],

  // KernelDeepDive.jsx에서 data.profiling_report로 접근
  profiling_report: {
    DRAM_대역폭_활용: "55.8%",
    SM_연산_효율: "High (Exp/Log)",
    레지스터_사용량: "Medium",
    워프_분기_효율: "96.5%",
    수치_안정성: "Stable (Sub-Max)",
  },

  // KernelDeepDive.jsx에서 data.analysis로 접근
  analysis:
    "CrossEntropy는 'Log-Softmax' 연산이 포함되어 있어 expf, logf 등의 초월 함수 비용이 높다. v2 구현은 Logits(입력)을 한 번만 읽고, Shared Memory Reduction을 통해 수치 안정성을 위한 Max 값과 Sum 값을 동시에 처리한다. Backward 패스에서는 메모리를 절약하기 위해 Softmax 값을 저장하지 않고 다시 계산(Recomputation)하는 전략을 취했다.",
};