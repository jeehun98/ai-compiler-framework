const tileCompatibleDetail = {
  overview: {
    title: "Method Overview",
    summary:
      "Tile-Compatible Compute는 연산을 작은 온칩 working set 안에서 수행하도록 재구성하여 shared memory, SRAM, 또는 L1 cache에 데이터가 머무는 동안 최대한 많은 연산을 수행하는 전략입니다.",
    problem:
      "연산 순서가 타일 친화적으로 구성되지 않으면 같은 데이터를 여러 번 global memory(HBM)에서 읽게 되고, 데이터 locality가 크게 떨어집니다.",
    property:
      "핵심은 computation ordering이 tile boundary 안에서 닫힐 수 있는지 여부입니다. reuse 가능한 데이터가 온칩에 머무는 동안 가능한 많은 연산을 완료해야 합니다.",
    impact:
      "resident tile 내부에서 데이터 재사용을 극대화하여 global memory 접근을 줄이고, compute 자원이 실제 성능으로 이어질 수 있는 조건을 만듭니다.",
  },

  theory: {
    title: "Math & Logic",
    body: [
      "모든 연산이 단순히 타일로 나뉠 수 있다고 해서 좋은 성능이 보장되는 것은 아닙니다.",
      "좋은 성능을 위해서는 dependency와 accumulation 순서가 tile 내부에서 닫혀 있어야 합니다.",
      "즉 partial result를 너무 일찍 외부 메모리로 내보내지 않고 local reuse가 충분히 일어날 수 있어야 합니다.",
      "이 문제는 단순한 분할이 아니라 tile-compatible schedule이 가능한지에 대한 구조적 조건과 관련됩니다.",
    ],
    bullets: [
      "Local data reuse 가능성",
      "Tile-closed dependency structure",
      "Partial accumulation locality",
      "Schedule-friendly compute ordering",
    ],
  },

  hardware: {
    title: "Physical Analysis",
    body: [
      "GPU에서는 shared memory, SRAM, 그리고 L1 cache의 크기가 제한되어 있으므로 tile 크기와 access pattern이 성능에 직접적인 영향을 줍니다.",
      "좋은 tile-compatible 구조는 working set이 온칩 메모리에 머무는 동안 반복적으로 재사용되도록 합니다.",
      "이를 통해 global memory(HBM) 접근을 줄이고 memory bandwidth pressure를 낮출 수 있습니다.",
      "GEMM, convolution, attention과 같은 많은 핵심 AI 커널이 이 원리에 기반하여 설계됩니다.",
    ],
    bullets: [
      "SRAM / shared memory residency",
      "Cache line reuse 증가",
      "HBM round-trip 감소",
      "Tile size와 occupancy 간 trade-off 존재",
    ],
  },

  compiler: {
    title: "MCIR Implementation",
    body: [
      "MCIR에서는 tile-compatibility를 단순한 schedule hint가 아니라 legality-bearing property로 표현하는 것이 중요합니다.",
      "연산이 tile boundary 내부에서 partial state를 유지하며 진행 가능한지, 그리고 필요한 working set이 온칩 메모리 제약을 만족하는지를 검사해야 합니다.",
      "이 조건이 만족되면 compiler는 tiled loop nest, local buffer placement, 그리고 fused compute schedule로 lowering할 수 있습니다.",
      "결과적으로 커널은 shared-memory resident execution 구조로 매핑될 수 있습니다.",
    ],
    bullets: [
      "Property: tile_compatible_compute",
      "Legality: working-set-fit / dependency closure",
      "Rewrite: naive loop → tiled schedule",
      "Kernel mapping: shared-memory resident kernel",
    ],
  },
};

export default tileCompatibleDetail;