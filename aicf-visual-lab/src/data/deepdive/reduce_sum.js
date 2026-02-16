export const reduceSumDeepDive = {
  id: "ReduceSum",

  kernel_evolution: [
    {
      version: "v1.0",
      tag: "베이스라인: 나이브 Column-wise",
      throughput: "1.7 ~ 102 GB/s ❌",
      description:
        "N=1(Global Sum)에서는 동시에 실행 가능한 블록이 1개 수준으로 병렬성이 붕괴하고, N=1024(Bias Grad)에서는 Row-major 데이터에 대한 Column 접근으로 코얼레싱이 깨져 대역폭이 크게 붕괴함. half2 벡터화는 일부 회복에만 기여.",
    },
  ],

  profiling_report: {
    기준_Copy_F32: "≈298 GB/s",
    Global_Sum_F32: "12.3 GB/s (≈4.1%)",
    Global_Sum_F16: "1.7 GB/s (≈0.6%)",
    Bias_Grad_F32: "29.6 GB/s (≈9.9%)",
    Bias_Grad_F16: "102.1 GB/s (≈34.3%)",
    주요_병목: "병렬성(grid) 붕괴 + Strided/Non-coalesced 접근",
    최적화_필요: "Global: 병렬 리덕션 / BiasGrad: 타일 기반 Coalesced Reduction",
  },

  analysis:
    "ReduceSum은 하드웨어 특성을 무시한 병렬화/메모리 접근이 얼마나 치명적인지 보여준다. N=1에서는 블록 1개 수준의 병렬성 붕괴가, N=1024에서는 Row-major 데이터에 대한 Column 접근이 코얼레싱을 깨뜨려 대역폭을 붕괴시킨다. 두 케이스는 성격이 달라 각각 병렬 리덕션과 타일 기반 coalesced reduction으로 분리 해결해야 한다.",
};
