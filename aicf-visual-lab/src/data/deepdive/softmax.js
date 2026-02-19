export const softmaxDeepDive = {
  id: "Softmax",

  // 1️⃣ 커널 진화 과정
  kernel_evolution: [
    {
      version: "v1.0 (Naïve)",
      tag: "Global Memory Pass",
      throughput: "N/A",
      description:
        "Max, Sum, Div 연산을 각각 별도의 커널로 실행. 중간 결과값을 전역 메모리에 쓰고 읽는 오버헤드로 인해 매우 느림.",
    },
    {
      version: "v1.1 (Fused Last-Dim)",
      tag: "Warp-Shuffle Fused",
      throughput: "145.47 GB/s (F32)",
      description:
        "하나의 블록이 한 행을 전담하여 Max/Sum/Apply를 단일 커널로 통합. Warp Shuffle을 사용하여 공유 메모리 병목을 제거함.",
    },
    {
      version: "v1.2 (Stability Optimized)",
      tag: "Online Softmax Ready",
      throughput: "136.66 GB/s (F16)",
      description:
        "FP16 입력에 대해 내부 계산을 FP32로 수행하여 수치적 안정성 확보. 지수 함수(exp) 계산 전 rmax 차감 로직이 완벽히 통합됨.",
    },
  ],

  // 2️⃣ 프로파일링 지표 (2048x8192 대형 텐서 기준)
  profiling_report: {
    최대_처리량: "145.47 GB/s (F32)",
    연산_복잡도: "Three-pass Fused",
    리덕션_방식: "Warp-Shuffle + Smem",
    수치_안정성: "Safe-max Applied",
    인플레이스_제한: "Strictly Blocked",
  },

  // 3️⃣ 핵심 분석 및 결론
  analysis:
    "Softmax는 Max 찾기, Exp 합계 계산, 정규화(Normalization)라는 3단계 과정을 거쳐야 합니다. 작성된 커널은 이 3단계를 단일 커널 내에서 'Three-pass'로 융합하여 메모리 I/O를 획기적으로 줄였습니다. 벤치마크 결과, F32에서 145 GB/s의 높은 대역폭을 달성하며 효율성을 입증했습니다. 특히, FP16 커널에서도 내부 리덕션은 FP32(Warp-Shuffle)로 수행하여 정밀도 손실을 최소화한 점이 돋보입니다. 다만, 데이터 규모가 커질수록 한 블록 내에서 행 데이터를 여러 번 순회(Loop)해야 하므로, 데이터 캐싱 전략이 성능의 핵심 변수로 작용합니다.",
};