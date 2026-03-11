const weightedReductionDetail = {
  overview: {
    title: "Method Overview",
    summary:
      "Streaming Weighted Reduction은 softmax-weighted sum과 같이 정규화된 weight가 필요한 reduction을 streaming 형태로 재구성하는 방법입니다.",
    problem:
      "일반적인 attention 구현은 score matrix(QKᵀ), softmax probability, weighted output을 단계적으로 materialize합니다. 이 과정에서 큰 intermediate tensor가 생성되고 HBM traffic이 크게 증가합니다.",
    property:
      "핵심은 rescaling invariance입니다. running max와 정규화된 denominator를 유지하면 이전 partial result를 새로운 scale에 맞춰 재정렬하며 누적할 수 있습니다.",
    impact:
      "이 구조를 이용하면 QKᵀ score matrix와 probability matrix를 전체 materialize하지 않고도 attention-like weighted reduction을 tile-streaming kernel로 구현할 수 있습니다.",
  },

  theory: {
    title: "Math & Logic",
    body: [
      "Weighted reduction이 어려운 이유는 단순한 합이 아니라 정규화된 weight가 필요하기 때문입니다.",
      "Softmax 기반 reduction에서는 exp(score)를 합산하기 전에 global normalization factor가 필요해 보입니다.",
      "하지만 online softmax 알고리즘은 running max와 renormalized denominator를 유지하여 streaming update를 가능하게 합니다.",
      "Accumulator 역시 새로운 max 기준으로 rescale하면 이전 block의 결과를 보존하면서 다음 block과 결합할 수 있습니다.",
    ],
    bullets: [
      "Running max tracking",
      "Renormalized denominator accumulation",
      "Accumulator rescaling",
      "FlashAttention 구조의 일반화",
    ],
  },

  hardware: {
    title: "Physical Analysis",
    body: [
      "이 구조의 가장 큰 하드웨어 이점은 score matrix와 probability matrix를 HBM에 저장하지 않는다는 점입니다.",
      "Q, K, V tile을 shared memory에 올리고 score 계산, 정규화, weighted accumulation을 하나의 kernel 내부에서 수행할 수 있습니다.",
      "이렇게 하면 intermediate tensor의 off-chip round trip이 크게 줄어듭니다.",
      "따라서 memory bandwidth가 병목인 attention 계열 연산에서 특히 효과적입니다.",
    ],
    bullets: [
      "Score matrix materialization 회피",
      "Probability write-back 제거",
      "Shared-memory tile residency 증가",
      "Attention kernel fusion 가능",
    ],
  },

  compiler: {
    title: "MCIR Implementation",
    body: [
      "MCIR에서는 이 구조를 weighted_streaming_reduction property로 모델링할 수 있습니다.",
      "컴파일러는 해당 reduction이 rescaling-invariant를 만족하는지, 그리고 accumulator state가 streaming merge 가능한지를 검사해야 합니다.",
      "조건이 만족되면 graph-level attention 패턴을 materialized intermediate 없이 streaming kernel 구조로 rewrite할 수 있습니다.",
      "lowering 단계에서는 tiled loop nest와 fused compute schedule을 통해 attention 연산을 단일 kernel로 매핑할 수 있습니다.",
    ],
    bullets: [
      "Property: weighted_streaming_reduction",
      "Legality: rescaling-safe / normalization invariant",
      "Rewrite: materialized attention → streaming attention",
      "Kernel mapping: tiled fused reduction kernel",
    ],
  },
};

export default weightedReductionDetail;