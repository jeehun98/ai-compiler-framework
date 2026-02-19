export const layerNormDeepDive = {
  id: "LayerNorm",

  // 1️⃣ 커널 진화 과정
  kernel_evolution: [
    {
      version: "v1.0 (Forward F32/F16)",
      tag: "Warp-Shuffle Reduction",
      throughput: "226.8 GB/s (F16)",
      description:
        "__shfl_down_sync를 활용하여 공유 메모리 접근 없이 워프 내에서 빠르게 합계를 계산. Row-wise 병렬화가 잘 이루어져 높은 대역폭을 달성함.",
    },
    {
      version: "v1.1 (Backward DX)",
      tag: "Two-Pass Calculation",
      throughput: "118.7 GB/s (F32)",
      description:
        "dx를 구하기 위해 sum(dy_hat)과 sum(dy_hat * xhat) 두 가지 리덕션이 필요함. 연산 복잡도가 증가하며 Forward 대비 처리량이 하락함.",
    },
    {
      version: "v1.2 (Backward DG/DB)",
      tag: "Column-wise Bottleneck",
      throughput: "25.7 GB/s (BERT shape)",
      description:
        "dgamma/dbeta 계산 시 Column-wise로 접근하면서 메모리 Coalescing이 깨짐. 특히 N이 작은 BERT 타입 형상에서 극심한 성능 저하 발생.",
    },
  ],

  // 2️⃣ 프로파일링 지표 (4096x4096 및 BERT-base 기준)
  profiling_report: {
    최대_처리량: "226.8 GB/s (F16 Fwd)",
    BERT_Bwd_효율: "25.7 GB/s (매우 낮음)",
    리덕션_알고리즘: "Warp-level Shuffle",
    메모리_레이아웃: "Row-major Contiguous",
    계산_정밀도: "F16/F32 Mixed",
  },

  // 3️⃣ 핵심 분석 및 결론
  analysis:
    "LayerNorm은 행(Row) 방향의 독립성이 강하여 Forward 패스에서는 Warp Shuffle을 통해 DRAM 대역폭의 한계치에 가까운 성능(>200 GB/s)을 냅니다. 하지만 Backward 패스, 특히 dgamma/dbeta를 구하는 과정에서 심각한 병목이 발생합니다. 현재 구현은 한 블록이 한 열(Column)을 담당하는데, 데이터가 Row-major로 저장되어 있어 읽기 시 Coalescing이 이루어지지 않습니다. 32768x768(BERT) 형상에서 처리량이 25.7 GB/s로 급락하는 것이 그 증거입니다. 성능을 개선하려면 dgamma/dbeta 계산 시 Block-wide Reduction을 넘어선 Global Atomic 또는 별도의 Transpose 커널 결합이 필요해 보입니다.",
};