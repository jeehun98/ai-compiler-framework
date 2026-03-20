export const batchNormDeepDive = {
  id: "BatchNorm",

  // 1️⃣ 커널 진화 과정
  kernel_evolution: [
    {
      version: "v1.0 (Inference)",
      tag: "Constant-Time Mapping",
      throughput: "194.87 GB/s",
      description:
        "이미 계산된 Running Stats를 사용하여 단순 Point-wise 연산 수행. 읽기(X)와 쓰기(Y)가 1:1로 매칭되어 FP16 환경에서 높은 효율을 보임.",
    },
    {
      version: "v1.1 (Forward Train)",
      tag: "Multi-Pass Atomic",
      throughput: "2.49 GB/s",
      description:
        "평균/분산 계산을 위해 전역 메모리에 대한 atomicAdd를 사용. 수천 개의 쓰레드가 동일한 [C] 주소에 경합(Contention)하면서 성능이 급격히 저하됨.",
    },
    {
      version: "v1.2 (Backward Train)",
      tag: "Gradient Aggregation",
      throughput: "3.89 GB/s",
      description:
        "dgamma, dbeta를 구하기 위해 다시 한번 Atomic Reduction 수행. Forward Train보다는 복잡하지만 유사한 경합 패턴을 보임.",
    },
  ],

  // 2️⃣ 프로파일링 지표 (32x64x128x128 테스트 결과 기준)
  profiling_report: {
    추론_처리량: "194.87 GB/s",
    학습_처리량: "2.49 GB/s",
    연산_정밀도: "Mixed (F16/F32)",
    리덕션_방식: "Global Atomic",
    메모리_패턴: "NCHW Contiguous",
  },

  // 3️⃣ 핵심 분석 및 결론
  analysis:
    "현재 BatchNorm 구현은 '정확성 우선(Correctness-first)' 전략을 택하고 있습니다. Inference 모드에서는 약 195 GB/s의 뛰어난 성능을 보여주지만, Training 모드에서는 전역 메모리에 대한 Atomic 연산 경합으로 인해 성능이 크게 제한됩니다. 특히 N*H*W(약 52만 개)의 요소가 단 64개의 채널 슬롯(C)에 동시 접근하려는 병목 현상이 관찰됩니다. 향후 성능 개선을 위해서는 Shared Memory를 활용한 Warp/Block-level Reduction을 도입하여 전역 메모리 원자 연산 횟수를 최소화하는 'Two-pass reduction' 최적화가 필요합니다.",
};