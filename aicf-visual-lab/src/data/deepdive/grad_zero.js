export const gradZeroDeepDive = {
  id: "GradZero",

  kernel_evolution: [
    {
      version: "v1.0",
      tag: "런타임 API 사용 (cudaMemsetAsync)",
      throughput: "315.2 GB/s (64MB)",
      description:
        "별도 커널 구현 없이 cudaMemsetAsync로 0 초기화를 수행. 큰 버퍼(64MB)에서 높은 DRAM 스트리밍 처리량이 관측되며, Memory Bound 커널의 기준점(Anchor)로 사용 가능.",
    },
  ],

  profiling_report: {
    _64MB_처리량: "315.2 GB/s",
    _32MB_처리량: "288.1 GB/s",
    특성: "바이트 단위 0 채우기(Write 중심)",
    해석: "버퍼가 작아질수록 오버헤드 비중 증가로 GB/s가 낮아질 수 있음",
  },

  analysis:
    "GradZero는 연산 최적화가 아니라 시스템의 메모리 스트리밍 처리량을 확인하는 기준점이다. 이 환경에서 64MB memset은 약 315 GB/s 수준으로 관측되며, 이후 Memory Bound 커널(Add, Adam 등)은 이 수치(또는 Copy baseline) 대비 비율로 평가하는 것이 가장 견고하다.",
};
