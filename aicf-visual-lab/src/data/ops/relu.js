// src/data/ops/relu.js

export const reluData = {
  id: "ReLU",
  category: "비선형 게이팅 / 반공간 정류 (Nonlinear Gating)",

  descriptions: {
    oneLine:
      "ReLU는 입력을 0 기준으로 양수 반공간은 보존하고 음수 반공간은 소거하는 output-local half-space gating operator입니다.",
    essence:
      "ReLU는 입력을 0을 경계로 나누어 음수 구간은 제거하고 양수 구간은 그대로 통과시키는 반공간 게이팅 연산입니다. 이를 통해 표현에 비선형성과 활성 희소성을 부여합니다.",
    strategy:
      "ReLU는 element-wise thresholding 구조를 가지므로 standalone pointwise kernel로도 실행 가능하지만, 실제로는 GEMM, Conv, BiasAdd 같은 선행 연산의 epilogue에 결합되는 lowering이 가장 자연스럽습니다. 핵심은 별도 메모리 왕복 없이 output-local gating을 수행하는 것입니다.",
    realization:
      "주로 fused epilogue 또는 branchless pointwise gating family로 연결되며, standalone kernel보다 결합된 realization이 더 자연스럽습니다. epilogue injection, branchless max, sparsity metadata generation의 상세 메커니즘은 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: "y_{i,j} = \\max(0, x_{i,j})",
    shapes: {
      x: "M x N",
      y: "M x N",
    },
    interpretation: {
      M: "샘플/행 축",
      N: "특징/채널 축",
      x: "비선형 게이팅 이전의 활성값",
      y: "정류(Rectified) 이후의 출력 활성값",
      "0": "활성/비활성 경계값",
    },
  },

  semantics: {
    thesis:
      "ReLU는 0을 기준으로 입력 공간을 두 개의 반공간으로 분할하여, 음수 영역은 비활성화하고 양수 영역은 보존하는 half-space gating operator입니다. 이 연산은 표현의 부호 구조를 활성 패턴으로 변환하며 후행 계산의 희소성 특성에 직접 영향을 줍니다.",
    axes: {
      M: {
        name: "Samples",
        role: "독립적으로 게이팅되는 데이터 행/샘플 축",
        description:
          "각 샘플 또는 행은 다른 행과 독립적으로 원소별 게이팅됩니다.",
      },
      N: {
        name: "Features",
        role: "각 원소가 독립적으로 판정되는 특징/채널 축",
        description:
          "각 feature 원소는 자신의 값만으로 활성/비활성 여부가 결정됩니다.",
      },
    },
  },

  propertyProfile: {
    associative_merge: {
      status: "not_applicable",
      score: 0.03,
      summary:
        "ReLU는 reduction merge 구조가 아니라 원소별 게이팅 연산입니다.",
      reason:
        "\\text{ReLU is an element-wise gating operator, not a reduction / merge operator}",
      allows: [],
    },

    local_accumulable: {
      status: "strong",
      score: 0.99,
      summary:
        "각 출력 원소는 동일 위치의 입력 하나만 보고 계산되므로 output-local realization이 극도로 강합니다.",
      reason:
        "y_{i,j} \\text{ depends only on } x_{i,j}, \\text{ making ReLU fully output-local}",
      allows: [
        "Output-Local Gating",
        "Pointwise Fusion",
        "Epilogue Injection",
      ],
    },

    tile_composable: {
      status: "strong",
      score: 0.98,
      summary:
        "입력/출력 domain을 어떤 타일로 나눠도 각 타일이 독립적으로 처리될 수 있습니다.",
      reason:
        "\\text{Any tiling over } (M,N) \\text{ preserves semantics because ReLU is element-wise independent}",
      allows: [
        "Block Tiling",
        "Vector Tile Apply",
        "Elementwise Partitioning",
      ],
    },

    order_rewritable: {
      status: "strong",
      score: 0.96,
      summary:
        "원소 간 의존성이 없어서 처리 순서 재배치가 자유롭습니다.",
      reason:
        "\\text{Element evaluation order may vary freely because each ReLU output is independent}",
      allows: [
        "Element Reordering",
        "Parallel Scheduling",
        "Vectorized Traversal",
      ],
    },

    schedule_invariant: {
      status: "strong",
      score: 0.97,
      summary:
        "warp/block/vector schedule 변화가 의미에 거의 영향을 주지 않는 대표적인 pointwise op입니다.",
      reason:
        "\\text{Execution schedule is highly flexible because gating is pointwise and stateless}",
      allows: [
        "Warp/Block Rescheduling",
        "Vectorized Schedule",
        "Fused Producer-Consumer Schedule",
      ],
    },

    representation_invariant: {
      status: "strong",
      score: 0.9,
      summary:
        "branchless max, predicate apply, vectorized layout 등 다양한 표현 방식으로 바꿔도 의미 보존이 쉽습니다.",
      reason:
        "\\text{Layout, vectorization, and branchless predicate realizations preserve ReLU semantics naturally}",
      allows: [
        "Branchless Max Realization",
        "Predicate-Based Gating",
        "Vectorized Load/Store",
      ],
    },

    rematerializable: {
      status: "strong",
      score: 0.86,
      summary:
        "upstream activation만 있으면 ReLU 출력은 매우 싸게 재계산할 수 있습니다.",
      reason:
        "y = \\max(0, x) \\text{ is trivially recomputable from the producer output } x",
      allows: [
        "Cheap Recompute",
        "Saved-Activation Elision",
        "Producer-Side Regeneration",
      ],
    },

    precision_relaxable: {
      status: "conditional",
      score: 0.74,
      summary:
        "threshold gating 자체는 단순하지만 0 근처 값에서는 표현 정밀도 변화가 경계 판정에 영향을 줄 수 있습니다.",
      reason:
        "\\text{ReLU is numerically simple, though values near zero can be sensitive to precision-induced sign changes}",
      allows: [
        "Mixed Precision Pointwise Apply",
        "Branchless Approximate Path",
      ],
    },

    domain_prunable: {
      status: "conditional",
      score: 0.67,
      summary:
        "음수 영역이 정확히 0으로 소거되므로 downstream skip, sparsity metadata 생성과 잘 맞습니다.",
      reason:
        "\\text{Negative half-space is exactly erased, enabling sparsity-aware downstream opportunities}",
      allows: [
        "Zero-Skipping Opportunity",
        "Sparsity Bitmask Generation",
        "Dead-Activation Analysis",
      ],
    },
  },

  opConstraints: [
    {
      id: "NONNEGATIVITY",
      name: "비음수 출력성 (Non-Negativity)",
      detail:
        "ReLU 출력은 항상 0 이상이어야 하며, 이 성질이 activation compression과 unsigned-friendly realization의 기초가 됩니다.",
      metric: "y_{i,j} \\ge 0",
      consequence: ["Activation Compression", "Unsigned-Friendly Realization"],
    },
    {
      id: "POSITIVE_IDENTITY",
      name: "양수 구간 항등성 (Positive Identity)",
      detail:
        "양수 입력은 값이 그대로 보존되어야 하므로 epilogue fusion에서도 positive half-space identity가 유지되어야 합니다.",
      metric: "x_{i,j} > 0 \\Rightarrow y_{i,j} = x_{i,j}",
      consequence: ["Epilogue Fusion", "Output-Local Gating"],
    },
    {
      id: "NEGATIVE_ERASURE",
      name: "음수 구간 소거성 (Negative Erasure)",
      detail:
        "음수 입력은 정확히 0으로 떨어져야 하며, 이 성질이 sparsity metadata와 skip opportunity의 근거가 됩니다.",
      metric: "x_{i,j} \\le 0 \\Rightarrow y_{i,j} = 0",
      consequence: ["Sparsity Bitmask", "Zero-Skipping Opportunity"],
    },
  ],

  downstreamConstraints: [
    {
      name: "Bias / GEMM / Conv Epilogue",
      rule:
        "\\text{후행 ReLU는 선행 연산의 output-local 결과에만 의존하므로 writeback 이전 epilogue로 결합 가능하다}",
      hint: "Epilogue fusion 우선",
    },
    {
      name: "Sparsity-Aware Execution",
      rule:
        "\\text{ReLU 이후 0 비율이 높아지면 후행 pointwise 또는 sparse-friendly 연산에서 skip opportunity가 생길 수 있다}",
      hint: "Activation sparsity metadata 활용 검토",
    },
    {
      name: "Dead Activation Regions",
      rule:
        "\\text{특정 채널 또는 경로가 장기간 } 0 \\text{ 출력에 머물면 표현 경로 활용도가 낮아질 수 있다}",
      hint: "Channel/path pruning 분석 후보",
    },
  ],

  lowering: {
    preferredFamily: "Fused Epilogue / Branchless Pointwise Gating",
    candidates: [
      "Standalone_Pointwise_ReLU",
      "Fused_Epilogue_ReLU",
      "Branchless_Vectorized_ReLU",
    ],
    chosen: {
      variant: "Fused_Epilogue_ReLU",
      summary:
        "ReLU는 출력 원소 하나에만 의존하는 output-local gating이므로, 선행 GEMM/Conv/BiasAdd의 epilogue에 직접 결합하는 realization이 가장 자연스럽습니다.",
      reason: [
        "\\text{출력 국소성(Output Locality): } y_{i,j} \\text{ 는 } x_{i,j} \\text{ 하나에만 의존하므로 standalone kernel보다 epilogue 결합이 자연스럽다}",
        "\\text{의미 보존 하의 결합: 선행 GEMM/Conv/BiasAdd 결과를 별도 메모리에 기록하기 전에 threshold gating을 적용할 수 있다}",
        "\\text{중간 버퍼 제거: standalone ReLU를 없애면 추가 load/store를 줄일 수 있다}",
      ],
      applied_rewrites: [
        "Epilogue Fusion",
        "Branchless Max Realization",
        "Optional Sparsity Metadata Generation",
      ],
    },
  },

  realizationSnapshot: {
    family: "Fused Epilogue / Branchless Pointwise Gating",
    highlights: [
      "Output-local threshold gating",
      "Branchless max / predicate apply",
      "Near-free when fused",
      "Optional sparsity metadata generation",
    ],
  },

  costModel: {
    semanticLoss:
      "\\mathcal{C}_{relu} = w_{pos} \\cdot \\Delta_{positive} + w_{neg} \\cdot \\Delta_{zeroing} + w_{gate} \\cdot \\Delta_{gating}",
    weights_hint: {
      default: {
        positive: 45.0,
        zeroing: 45.0,
        gating: 10.0,
      },
    },
    metrics: {
      positive_identity_consistency: "High",
      negative_zeroing_consistency: "High",
      epilogue_affinity: "Strong",
    },
  },

  performance: {
    latency: {
      pytorch: 0.05,
      torch_compile: 0.02,
      ours: 0.0,
    },
  },

  gatingFamily: {
    gatingType: "half-space-threshold",
    threshold: 0.0,
    outputLocal: true,
    sparseOutputPossible: true,
    epilogueFriendly: true,
    branchlessFriendly: true,
  },

  cudaCode: `// AICF: Fused ReLU (Epilogue)
// Inside GEMM / Conv / BiasAdd epilogue:
float acc = ...;          // upstream result
acc = fmaxf(acc, 0.0f);   // ReLU gating
*output = acc;            // final writeback`,
};