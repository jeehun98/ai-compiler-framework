export const adamStepDeepDive = {
  id: "AdamStep",

  // 1️⃣ 커널 진화 과정: 기본 구현에서 최적화까지
  kernel_evolution: [
    {
      version: "v1.0 (PyTorch Ref)",
      tag: "Python Reference",
      throughput: "N/A",
      description:
        "Torch 연산자들의 조합으로 구현. 중간 결과값(m_hat, v_hat 등) 생성을 위해 다수의 임시 텐서가 할당되며 메모리 I/O가 매우 빈번함.",
    },
    {
      version: "v2.0 (OOP)",
      tag: "Out-of-Place CUDA",
      throughput: "231.38 GB/s",
      description:
        "단일 커널로 퓨전(Fusion)되어 임시 텐서 생성을 제거함. 다만, P에서 Pout으로의 별도 cudaMemcpyAsync 복사 오버헤드가 발생함.",
    },
    {
      version: "v2.1 (In-Place)",
      tag: "In-Place Optimized",
      throughput: "292.00 GB/s",
      description:
        "데이터 복사 없이 기존 메모리 주소(P, M, V)에서 즉시 업데이트를 수행. OOP 대비 약 26%의 대역폭 향상을 보이며 이론적 최대치에 근접함.",
    },
  ],

  // 2️⃣ 프로파일링 지표 (4096x4096 테스트 결과 기준)
  profiling_report: {
    메모리_대역폭_효율: "292.0 GB/s",
    연산_지연_시간: "1.609 ms",
    데이터_이동량: "~448 MB",
    I_O_패턴: "7-way Access",
    커널_퓨전_상태: "Fully Fused",
  },

  // 3️⃣ 핵심 분석 및 결론
  analysis:
    "AdamStep 커널은 전형적인 Memory-Bound 연산입니다. 한 번의 연산을 위해 4개 배열(P, G, M, V)을 읽고 3개 배열(P, M, V)을 써야 하는 '7-way Memory Access' 구조를 가집니다. 테스트 결과, In-Place 방식이 OOP 방식보다 빠른 이유는 불필요한 cudaMemcpy 호출을 제거하고 데이터 지역성(Locality)을 극대화했기 때문입니다. 현재 구현된 v2 커널은 Grid-Stride Loop를 통해 다양한 Shape에 유연하게 대응하며, 256 쓰레드 배치를 통해 메모리 Coalescing을 최적으로 활용하고 있습니다.",
};