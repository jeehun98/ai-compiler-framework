javascript
export const adamStepDeepDive = {
  id: "AdamStep",

  // KernelDeepDive.jsx에서 data.kernel_evolution으로 접근
  kernel_evolution: [
    {
      version: "v1.0 (In-Place)",
      tag: "단일 커널 퓨전",
      throughput: "310.1 GB/s",
      description:
        "Momentum, Variance, Parameter 갱신을 하나의 커널로 통합. 28 Bytes의 필수 데이터 이동만 수행하여 하드웨어 대역폭 한계(Saturation)에 도달함.",
    },
    {
      version: "v1.1 (OOP)",
      tag: "Out-of-Place (Safe Mode)",
      throughput: "235.3 GB/s",
      description:
        "입출력 텐서가 다를 경우 정합성을 위해 선행 복사(Memcpy) 수행. 요소당 8 Bytes의 추가 이동 비용 발생으로 대역폭 효율 하락.",
    },
  ],

  // KernelDeepDive.jsx에서 data.profiling_report로 접근
  profiling_report: {
    유효_메모리_대역폭: "310.1 GB/s",
    하드웨어_포화도: "98.7%",
    요소당_데이터_이동: "28 Bytes",
    병목_지점: "DRAM Interface",
  },

  // KernelDeepDive.jsx에서 data.analysis로 접근
  analysis:
    "Adam 최적화의 핵심은 수식을 줄이는 것이 아니라, 28 Bytes를 한 번에 처리하는 구조를 만드는 것이다. 커널 퓨전을 통해 메모리 재접근을 제거함으로써, 복잡한 수식에도 불구하고 단순 복사(Copy)에 준하는 대역폭 성능을 달성했다.",
};