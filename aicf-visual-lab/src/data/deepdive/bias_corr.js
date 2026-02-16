export const biasCorrDeepDive = {
  id: "BiasCorr",

  kernel_evolution: [
    {
      version: "v0.1",
      tag: "스칼라 헬퍼 커널",
      throughput: "런치 지연 지배 (~수 µs)",
      description:
        "Adam Optimizer의 보정 계수를 계산하는 초경량 커널. 연산 부하는 매우 작고, Host-Device 동기화를 피하기 위해 GPU에서 직접 계산한다.",
    },
  ],

  profiling_report: {
    지연_특성: "대부분이 커널 런치 오버헤드",
    메모리_사용: "스칼라 Read/Write (무시 가능)",
    안전_처리: "t=0 입력 시 t=1로 Clamp (Div/0 회피)",
    사용_의도: "CUDA Graph/비동기 파이프라인 유지",
  },

  analysis:
    "이 커널은 성능 지표를 올리기 위한 것이 아니라 파이프라인 연속성을 위해 존재한다. Step 텐서가 GPU에 있을 때 이를 CPU로 가져오지 않고 즉시 계산함으로써, CPU-GPU 동기화 비용을 제거하고 CUDA Graph 캡처/리플레이 흐름을 유지한다.",
};
