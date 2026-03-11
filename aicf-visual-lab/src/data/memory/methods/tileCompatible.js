const tileCompatibleDetail = {
  overview: {
    title: "Method Overview",
    summary:
      "Tile-Compatible Compute는 연산을 작은 온칩 working set 안에서 닫히도록 재구성하여 SRAM/L1/shared memory residency를 유지하는 전략입니다.",
    problem:
      "연산 순서가 타일 친화적으로 구성되지 않으면 같은 데이터를 여러 번 HBM에서 불러오게 되고 locality가 무너집니다.",
    property:
      "핵심은 computation ordering이 tile boundary 안에서 닫힐 수 있는지입니다. reuse 가능한 데이터가 온칩에 머무는 동안 최대한 많은 연산을 끝내야 합니다.",
    impact:
      "resident tile 안에서 reuse를 극대화하여 memory wall을 완화하고, compute throughput이 실제로 발휘될 수 있는 조건을 만듭니다.",
  },
  theory: {
    title: "Math & Logic",
    body: [
      "모든 연산이 타일링 가능하다고 해도, 실제로 좋은 성능을 내려면 dependency와 accumulation 순서가 tile 안에서 닫혀야 합니다.",
      "즉 partial result를 너무 일찍 외부로 내보내지 않고, local reuse를 충분히 수행할 수 있어야 합니다.",
      "이는 단순 분할이 아니라 tile-compatible schedule 여부의 문제입니다.",
    ],
    bullets: [
      "Local reuse 가능성",
      "Tile-closed dependency",
      "Partial accumulation locality",
      "Schedule-friendly compute ordering",
    ],
  },
  hardware: {
    title: "Physical Analysis",
    body: [
      "하드웨어에서는 shared memory, SRAM, L1 cache 크기가 제한되어 있으므로 tile 크기와 access pattern이 직접 성능을 좌우합니다.",
      "좋은 tile-compatible 구조는 working set이 온칩에 머무는 동안 반복 재사용되며, global memory round trip을 최소화합니다.",
      "GEMM, convolution, attention 모두 이 원리에 크게 의존합니다.",
    ],
    bullets: [
      "SRAM/shared memory residency",
      "Cache line reuse 증가",
      "HBM round trip 감소",
      "Occupancy와 tile size trade-off 존재",
    ],
  },
  compiler: {
    title: "MCIR Implementation",
    body: [
      "MCIR에서는 tile-compatibility를 단순 schedule hint가 아니라 legality-bearing property로 보는 것이 중요합니다.",
      "연산이 tile boundary 안에서 partial state를 유지하며 진행 가능한지, 필요한 working set이 온칩 자원 제한을 만족하는지 검사해야 합니다.",
      "성립하면 compiler는 tiled loop nest, local buffer placement, fused schedule로 lowering할 수 있습니다.",
    ],
    bullets: [
      "Property: tile_compatible_compute",
      "Legality: working-set-fit / dependency closure",
      "Rewrite: naive loop -> tiled schedule",
      "Kernel mapping: shared-memory resident kernel",
    ],
  },
};

export default tileCompatibleDetail;