export const biasAddDeepDive = {
  id: "BiasAdd",

  kernel_evolution: [
    {
      version: "v1.0",
      tag: "F16 스칼라 브로드캐스팅 (Naive)",
      throughput: "219.4 GB/s",
      description:
        "요소마다 bias 인덱싱(모듈러/주소 계산)이 발생해 명령 발행 오버헤드가 커지고 ILP가 부족해져, 메모리 파이프를 충분히 포화시키지 못함.",
    },
    {
      version: "v2.0",
      tag: "F16 벡터화 (half2)",
      throughput: "278.4 GB/s",
      description:
        "half2 벡터화로 2개 요소당 1회만 인덱싱/주소 계산을 수행해 오버헤드를 절감. 그 결과 유효 메모리 대역폭을 크게 회복(+27%).",
    },
  ],

  profiling_report: {
    유효_메모리_대역폭: "278.4 GB/s",
    성능_향상폭: "+27% (vs Naive)",
    병목_특성: "주소 계산(인덱싱) + 메모리 파이프 포화 부족",
    측정_정의: "유효 대역폭 = Read(Y)+Write(Out) 기준(2×), bias read는 비교 목적상 제외",
  },

  analysis:
    "BiasAdd 같은 브로드캐스팅 커널에서는 데이터 이동량만큼이나 '어디를 읽을지 계산하는 비용(주소 계산/인덱싱)'이 병목이 될 수 있다. half2 벡터화는 주소 계산 빈도를 절반으로 낮춰 명령 발행 오버헤드를 줄이고, 메모리 파이프를 더 잘 포화시키는 핵심 전략이다.",
};
