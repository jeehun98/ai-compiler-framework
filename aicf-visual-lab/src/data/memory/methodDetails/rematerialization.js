const rematerializationDetail = {
  overview: {
    title: "Method Overview",
    summary:
      "Re-materializable Intermediate는 저장 비용이 재계산 비용보다 더 비싼 intermediate를 즉시 재생성하는 전략입니다.",
    problem:
      "모든 중간값을 저장하면 VRAM 사용량과 HBM traffic이 커지고, 특히 activation-heavy 구조에서는 저장 그 자체가 병목이 됩니다.",
    property:
      "핵심은 어떤 intermediate가 cheap-to-recompute인지 판별하는 것입니다. 연산량은 작고 저장 비용이 큰 경우 rematerialization이 유리합니다.",
    impact:
      "VRAM usage를 줄이고 peak memory pressure를 낮추며, 일부 구간에서는 bandwidth 절감으로 실제 속도에도 이점이 생깁니다.",
  },
  theory: {
    title: "Math & Logic",
    body: [
      "모든 intermediate가 동일한 가치를 가지는 것은 아닙니다.",
      "어떤 값은 expensive result라 저장해야 하지만, 어떤 값은 입력 몇 개만 있으면 빠르게 다시 만들 수 있습니다.",
      "따라서 computation graph에서는 store 대상으로 볼지 recompute 대상으로 볼지 구분하는 cost model이 필요합니다.",
    ],
    bullets: [
      "Cheap-to-recompute vs expensive-to-store",
      "Lifetime 축소",
      "Peak memory 감소",
      "Checkpointing과 연결 가능",
    ],
  },
  hardware: {
    title: "Physical Analysis",
    body: [
      "하드웨어적으로는 HBM 접근이 비싸고 ALU/FMA 연산이 상대적으로 남는 상황에서 특히 유리합니다.",
      "즉 메모리 병목이 큰 구간에서는 intermediate load/store를 줄이기 위해 약간의 추가 연산을 감수하는 편이 더 낫습니다.",
      "이 전략은 activation checkpointing, fused epilogue, transient tensor 제거와 강하게 연결됩니다.",
    ],
    bullets: [
      "HBM read/write 감소",
      "추가 연산으로 bandwidth 절약",
      "Peak VRAM 감소",
      "Memory-bound kernel에 유리",
    ],
  },
  compiler: {
    title: "MCIR Implementation",
    body: [
      "MCIR에서는 intermediate tensor를 must-materialize와 rematerializable로 나누는 property가 필요합니다.",
      "legality는 재계산이 부작용 없이 동일 의미를 재현하는지, 그리고 재계산 비용이 허용 범위인지로 결정됩니다.",
      "lowering에서는 store/load edge를 제거하고 producer subgraph를 필요한 지점에 다시 삽입하는 방식으로 구현할 수 있습니다.",
    ],
    bullets: [
      "Property: rematerializable_intermediate",
      "Legality: pure / deterministic / cheap-enough",
      "Rewrite: materialize edge -> recompute edge",
      "Kernel mapping: fused local recompute",
    ],
  },
};

export default rematerializationDetail;