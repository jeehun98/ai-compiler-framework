const rematerializationDetail = {
  overview: {
    title: "Method Overview",
    summary:
      "Rematerialization은 intermediate 값을 저장하는 대신 필요할 때 다시 계산하는 전략입니다. 저장 비용보다 재계산 비용이 작은 경우 메모리 사용량을 줄일 수 있습니다.",
    problem:
      "모든 intermediate tensor를 저장하면 VRAM 사용량과 HBM traffic이 증가합니다. 특히 activation-heavy 모델에서는 intermediate storage 자체가 메모리 병목을 만들 수 있습니다.",
    property:
      "핵심은 어떤 intermediate가 cheap-to-recompute인지 판단하는 것입니다. 연산 비용이 작고 저장 비용이 큰 경우 rematerialization이 효과적입니다.",
    impact:
      "VRAM 사용량과 peak memory pressure를 낮추며, 일부 상황에서는 memory traffic 감소로 실제 실행 속도에도 이점이 생길 수 있습니다.",
  },

  theory: {
    title: "Math & Logic",
    body: [
      "모든 intermediate가 동일한 비용 구조를 가지는 것은 아닙니다.",
      "어떤 값은 계산 비용이 크기 때문에 저장하는 것이 유리하지만, 어떤 값은 입력만 있으면 빠르게 다시 계산할 수 있습니다.",
      "따라서 computation graph에서는 intermediate를 materialize할지 recompute할지 결정하는 cost model이 필요합니다.",
      "이 선택은 메모리 사용량과 연산량 사이의 tradeoff를 조정하는 과정입니다.",
    ],
    bullets: [
      "Cheap-to-recompute vs expensive-to-store",
      "Intermediate lifetime 축소",
      "Peak memory usage 감소",
      "Activation checkpointing과 연결",
    ],
  },

  hardware: {
    title: "Physical Analysis",
    body: [
      "GPU에서는 global memory(HBM) 접근이 ALU/FMA 연산보다 훨씬 비싼 경우가 많습니다.",
      "따라서 memory-bound 구간에서는 일부 연산을 다시 수행하는 것이 intermediate load/store를 반복하는 것보다 효율적일 수 있습니다.",
      "rematerialization은 이러한 특성을 이용하여 memory traffic을 줄이고 compute 자원을 더 적극적으로 활용합니다.",
      "이 전략은 activation checkpointing, fused epilogue, transient tensor 제거와 같은 최적화와도 밀접하게 연결됩니다.",
    ],
    bullets: [
      "HBM read/write 감소",
      "추가 연산으로 memory traffic 절약",
      "Peak VRAM usage 감소",
      "Memory-bound kernel에서 효과적",
    ],
  },

  compiler: {
    title: "MCIR Implementation",
    body: [
      "MCIR에서는 intermediate tensor를 must-materialize와 rematerializable로 구분하는 property가 필요합니다.",
      "legality는 해당 연산이 pure하고 deterministic하며 재계산 비용이 허용 가능한지에 따라 결정됩니다.",
      "graph rewrite 단계에서는 materialize edge를 제거하고 producer subgraph를 필요한 위치에서 다시 실행하도록 변환합니다.",
      "lowering 단계에서는 recompute가 tile-local 또는 fused kernel 내부에서 수행되도록 배치할 수 있습니다.",
    ],
    bullets: [
      "Property: rematerializable_intermediate",
      "Legality: pure / deterministic / cheap-enough",
      "Rewrite: materialize edge → recompute edge",
      "Kernel mapping: fused local recomputation",
    ],
  },
};

export default rematerializationDetail;