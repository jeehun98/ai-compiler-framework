export const biasAddDeepDive = {
  id: "BiasAdd",

  // 1️⃣ 커널 진화 과정: 데이터 타입과 벡터화에 따른 성능 변화
  kernel_evolution: [
    {
      version: "v1.0 (F32)",
      tag: "Standard FP32",
      throughput: "278.17 GB/s",
      description:
        "가장 기본적인 구현. 단일 정밀도 부동소수점을 사용하여 연산하며, 메모리 대역폭을 안정적으로 활용함.",
    },
    {
      version: "v1.1 (F16 Naive)",
      tag: "FP16 Scalar",
      throughput: "211.97 GB/s",
      description:
        "데이터 크기를 절반으로 줄였으나, 홀수 크기(Odd N)의 Shape으로 인해 벡터화 최적화를 적용하지 못한 사례. 메모리 접근 효율이 상대적으로 낮음.",
    },
    {
      version: "v1.2 (F16 Vec2)",
      tag: "FP16 Vectorized (half2)",
      throughput: "263.22 GB/s",
      description:
        "짝수 크기(Even N)와 4B 정렬을 만족하여 half2 명령어를 사용. Naive FP16 대비 처리량이 약 24% 향상됨.",
    },
  ],

  // 2️⃣ 프로파일링 지표 (4096x4096 테스트 결과 기준)
  profiling_report: {
    최대_처리량: "278.17 GB/s (F32)",
    벡터화_가속: "1.24x (vs Naive)",
    메모리_패턴: "Broadcast Read",
    최적_정렬: "4-byte Aligned",
    커널_선택_로직: "Dynamic Dispatch",
  },

  // 3️⃣ 핵심 분석 및 결론
  analysis:
    "BiasAdd 커널은 연산량 대비 메모리 이동량이 많은 전형적인 Memory-bound 연산입니다. 특히 Bias 값은 상대적으로 크기가 작아 L2 캐시에 상주하므로, 성능의 관건은 입력 Y를 읽고 출력 Out을 쓰는 과정에 있습니다. 본 구현의 백미는 'F16-Vec2' 최적화입니다. __half2 형태의 2-way 벡터 연산을 적용함으로써 메모리 명령어(Load/Store) 횟수를 절반으로 줄여 지연 시간을 단축했습니다. 다만, F32가 F16보다 처리량이 높게 측정된 점은 데이터 크기가 커짐에 따라 GPU의 메모리 컨트롤러가 대용량 버스트(Burst) 전송을 더 효율적으로 처리했기 때문으로 분석됩니다.",
};