const onlineNormDetail = {
  overview: {
    title: "Method Overview",
    summary:
      "Online Reducible Norm은 평균과 분산 계산을 multi-pass reduction이 아닌 streaming reduction으로 바꾸는 방식입니다.",
    problem:
      "기존 정규화 계열 계산은 mean pass와 variance pass를 분리하면서 입력을 여러 번 읽게 되고, 이는 HBM bandwidth를 낭비합니다.",
    property:
      "핵심은 통계량이 결합 가능한 상태(state)로 표현된다는 점입니다. Welford state는 partial merge가 가능하므로 tile 단위 병렬 reduction과 streaming accumulation 모두에 적합합니다.",
    impact:
      "입력 재방문 횟수를 줄이고 intermediate traffic을 낮춰 normalization 계열 연산을 memory-bound bottleneck에서 더 유리하게 만듭니다.",
  },
  theory: {
    title: "Math & Logic",
    body: [
      "평균과 분산은 단순히 전체 벡터를 모두 본 뒤 계산해야 하는 값처럼 보이지만, 실제로는 merge 가능한 상태로 표현할 수 있습니다.",
      "Welford 알고리즘은 count, mean, M2를 유지하면서 새 샘플이 들어올 때마다 통계량을 업데이트합니다.",
      "이 상태 표현은 두 개의 partial segment를 다시 합칠 수 있으므로, single-pass streaming뿐 아니라 block-wise parallel reduction에도 잘 맞습니다.",
      "즉, norm statistic은 full materialization 대상이 아니라 reducible state로 다룰 수 있습니다.",
    ],
    bullets: [
      "Reducer state: (count, mean, M2)",
      "Associative-style merge 가능",
      "Single-pass statistic accumulation",
      "Multi-pass HBM reread 제거 가능",
    ],
  },
  hardware: {
    title: "Physical Analysis",
    body: [
      "하드웨어 관점에서 가장 중요한 점은 같은 activation을 여러 번 다시 읽지 않는다는 것입니다.",
      "tile 내부에서 partial statistics를 register/shared memory에 유지하고 block reduction 뒤 최종 norm factor만 확정하면 됩니다.",
      "입력 전체를 두세 번 순회하는 방식보다 memory traffic이 감소하며, bandwidth pressure가 큰 구간일수록 효과가 커집니다.",
    ],
    bullets: [
      "HBM reread 감소",
      "Shared memory / register accumulation",
      "Reduction tree와 결합 쉬움",
      "Normalization kernel fusion 기반 제공",
    ],
  },
  compiler: {
    title: "MCIR Implementation",
    body: [
      "MCIR에서는 이 기법을 단순한 mean/var op 조합이 아니라 reducible-statistic property로 표현하는 것이 중요합니다.",
      "핵심 legality는 통계량이 merge-safe state로 분해 가능한지 여부입니다.",
      "lowering 단계에서는 tile-local accumulation + hierarchical merge + final normalization scale application 형태로 내릴 수 있습니다.",
    ],
    bullets: [
      "Property: reducible_state(statistics)",
      "Legality: associative merge / stable update",
      "Rewrite: multi-pass norm -> online stat reduction",
      "Kernel mapping: block reduction + final apply",
    ],
  },
};

export default onlineNormDetail;