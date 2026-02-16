export const gemmDeepDive = {
  id: "GEMM",

  // KernelDeepDive.jsx에서 data.kernel_evolution으로 접근
  kernel_evolution: [
    {
      version: "v1.0",
      tag: "순수 기본 구현",
      throughput: "245.2 GFLOPS",
      description:
        "전역 메모리에서 직접 데이터를 읽는 방식. 메모리 병목이 심각하여 연산 유닛이 대부분 유휴 상태로 남아 있음.",
    },
    {
      version: "v2.0",
      tag: "공유 메모리 타일링",
      throughput: "1,420.5 GFLOPS",
      description:
        "32×32 타일링 적용. 데이터 재사용성을 높여 DRAM 대역폭 요구량을 크게 감소시킴.",
    },
    {
      version: "v3.0",
      tag: "텐서코어 + 벡터화 입출력",
      throughput: "3,188.9 GFLOPS",
      description:
        "Ampere WMMA API 사용 및 Float4 벡터 로드 적용. 공유 메모리 뱅크 충돌을 완화하기 위한 패딩 기법 도입.",
    },
  ],

  // KernelDeepDive.jsx에서 data.profiling_report로 접근
  profiling_report: {
    SM_점유율: "92.4%",
    L1_캐시_적중률: "94.1%",
    텐서코어_활용률: "88.5%",
    DRAM_처리율: "74.2%",
    워프_실행_효율: "98.2%",
  },

  // KernelDeepDive.jsx에서 data.analysis로 접근
  analysis:
    "공유 메모리에 타일을 적재할 때 Stride-aware Padding을 적용하여 32개 뱅크 접근 충돌을 줄였다. 그 결과 v2 대비 지연 시간을 약 15% 추가 단축하였다.",
};
