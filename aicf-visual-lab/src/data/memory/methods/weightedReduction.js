const weightedReductionDetail = {
  overview: {
    title: "Method Overview",
    summary:
      "Streaming Weighted Reduction은 softmax-weighted sum처럼 weight가 동적으로 정규화되는 reduction을 스트리밍 가능 형태로 바꾸는 방식입니다.",
    problem:
      "일반적인 attention은 score matrix, softmax probability, weighted output을 단계적으로 materialize하기 쉽고, 이 과정에서 거대한 HBM traffic이 발생합니다.",
    property:
      "핵심은 rescaling invariant입니다. running max와 normalized denominator를 유지하면 과거 partial result를 새로운 scale에 맞춰 재정렬하며 누적할 수 있습니다.",
    impact:
      "QK^T와 probability matrix를 전부 저장하지 않고도 attention-like weighted reduction을 tile-streaming kernel로 바꿀 수 있습니다.",
  },
  theory: {
    title: "Math & Logic",
    body: [
      "Weighted reduction이 어려운 이유는 단순 합이 아니라 정규화된 weight가 필요하기 때문입니다.",
      "하지만 online softmax 계열 알고리즘은 running max와 renormalized sum을 유지하여 streaming update를 가능하게 만듭니다.",
      "partial accumulator 역시 새로운 max 기준으로 rescale하면 이전 block의 결과를 보존한 채 다음 block과 합칠 수 있습니다.",
    ],
    bullets: [
      "Running max 유지",
      "Renormalized denominator 누적",
      "Weighted accumulator rescaling",
      "FlashAttention 핵심 구조 일반화",
    ],
  },
  hardware: {
    title: "Physical Analysis",
    body: [
      "이 구조의 하드웨어 이점은 중간 score/probability matrix를 HBM에 저장하지 않는 데 있습니다.",
      "Q, K, V tile을 shared memory에 올리고, score 계산-정규화-누적을 같은 커널 안에서 끝내면 off-chip round trip이 크게 줄어듭니다.",
      "즉 compute보다 memory movement가 더 큰 병목일 때 특히 강력합니다.",
    ],
    bullets: [
      "Score matrix materialization 회피",
      "Probability write-back 제거",
      "Shared-memory residency 증가",
      "Attention-like kernel fusion 가능",
    ],
  },
  compiler: {
    title: "MCIR Implementation",
    body: [
      "MCIR에서는 이를 weighted-streaming-reduction property로 모델링할 수 있습니다.",
      "compiler는 해당 reduction이 rescaling-invariant를 만족하는지, accumulator state가 streaming merge 가능한지를 검사해야 합니다.",
      "성립하면 graph-level attention 패턴을 tile-streaming fused kernel로 rewrite할 수 있습니다.",
    ],
    bullets: [
      "Property: weighted_streaming_reduction",
      "Legality: rescaling-safe / normalization invariant",
      "Rewrite: materialized attention -> streaming attention",
      "Kernel mapping: tiled fused reduction kernel",
    ],
  },
};

export default weightedReductionDetail;