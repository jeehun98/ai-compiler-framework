export const layernormDeepDive = {
  id: "LayerNorm",

  kernel_evolution: [
    {
      version: "v1.0",
      tag: "베이스라인: FWD 스칼라 / BWD 스트라이드",
      throughput: "FWD: 235.7 GB/s, BWD(최악): 25.7 GB/s ❌",
      description:
        "FWD는 스칼라 로드/스토어 및 리덕션 비용으로 대역폭 포화에 실패했고, BWD는 Row-major 데이터에 대해 Column-wise reduction을 나이브하게 수행해 Non-coalesced(스트라이드) 접근이 발생하여 유효 대역폭이 붕괴함.",
    },
  ],

  profiling_report: {
    FWD_대역폭: "235.7 GB/s (4096×4096, F16)",
    BWD_대역폭_1: "88.4 GB/s (4096×4096, F16)",
    BWD_대역폭_2: "25.7 GB/s (32768×768, F16)",
    주요_병목: "BWD Non-coalesced / Strided Memory Access",
    최적화_우선순위: "BWD 타일 기반 Coalesced Reduction → FWD 벡터화",
  },

  analysis:
    "LayerNorm BWD는 dgamma/dbeta를 Column-wise로 누적하는 과정에서 Row-major 데이터에 대한 strided 접근이 발생해 유효 대역폭이 크게 붕괴한다(최악 25.7 GB/s). 해결을 위해서는 타일을 가로로 coalesced load한 뒤 warp/shmem reduction으로 세로 합을 누적하는 구조로 바꾸는 것이 필수이며, FWD는 half2/float4 벡터화와 리덕션 최적화로 280~300 GB/s 구간을 목표로 한다.",
};
