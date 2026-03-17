// src/data/relu.js

export const reluData = {
  id: "ReLU",
  category: "비선형 게이팅 / 반공간 정류 (Nonlinear Gating)",

  descriptions: {
    essence:
      "ReLU는 입력을 0을 경계로 나누어 음수 구간은 제거하고 양수 구간은 그대로 통과시키는 반공간 게이팅 연산입니다. 이를 통해 표현에 비선형성과 활성 희소성을 부여합니다.",
    strategy:
      "ReLU는 element-wise thresholding 구조를 가지므로 standalone pointwise kernel로도 실행 가능하지만, 실제로는 GEMM, Conv, BiasAdd 같은 선행 연산의 epilogue에 결합되는 lowering이 가장 자연스럽습니다. 핵심은 별도 메모리 왕복 없이 output-local gating을 수행하는 것입니다.",
    hardware:
      "이 연산은 주로 pointwise max 또는 predicate-based gating family로 연결되며, 실제 epilogue injection, branchless max, sparsity metadata 생성 여부는 Deep Dive 계층에서 다룹니다.",
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
      M: { name: "Samples", role: "독립적으로 게이팅되는 데이터 행/샘플 축" },
      N: { name: "Features", role: "각 원소가 독립적으로 판정되는 특징/채널 축" },
    },

    invariants: [
      {
        id: "INV_NONNEGATIVITY",
        name: "비음수 출력성 (Non-Negativity)",
        metric: "y_{i,j} \\ge 0",
        threshold: "Strict",
        allows: ["Activation Compression", "Unsigned-Friendly Realization"],
      },
      {
        id: "INV_POSITIVE_IDENTITY",
        name: "양수 구간 항등성 (Positive Identity)",
        metric: "x_{i,j} > 0 \\Rightarrow y_{i,j} = x_{i,j}",
        threshold: "Exact identity on positive half-space",
        allows: ["Epilogue Fusion", "Output-Local Gating"],
      },
      {
        id: "INV_NEGATIVE_ERASURE",
        name: "음수 구간 소거성 (Negative Erasure)",
        metric: "x_{i,j} \\le 0 \\Rightarrow y_{i,j} = 0",
        threshold: "Exact zeroing on negative half-space",
        allows: ["Sparsity Bitmask", "Zero-Skipping Opportunity"],
      },
    ],

    sensitivity: {
      downstream: [
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
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_Epilogue_ReLU",
      reason: [
        "\\text{출력 국소성(Output Locality): } y_{i,j} \\text{ 는 } x_{i,j} \\text{ 하나에만 의존하므로 standalone kernel보다 epilogue 결합이 자연스럽다}",
        "\\text{의미 보존 하의 결합: 선행 GEMM/Conv/BiasAdd 결과를 별도 메모리에 기록하기 전에 threshold gating을 적용할 수 있다}",
        "\\text{중간 버퍼 제거: standalone ReLU를 없애면 추가 load/store를 줄일 수 있다}",
        "\\text{따라서 } \\texttt{Fused\\_Epilogue\\_ReLU} \\text{ family가 적합하다}",
      ],
      applied_rewrites: [
        "Epilogue Fusion",
        "Branchless Max Realization",
        "Optional Sparsity Metadata Generation",
      ],
    },
  },

  kernel: {
    strategy: "Branchless Pointwise Gating",
    details: [
      {
        technique: "Epilogue Fusion",
        semantic_link: "선행 연산 결과의 writeback 직전에 threshold gating 적용",
      },
      {
        technique: "Branchless Max / Predicate Apply",
        semantic_link: "원소별 반공간 판정을 분기 없이 수행",
      },
      {
        technique: "Optional Sparsity Bitmask",
        semantic_link: "비활성 위치 정보를 후행 연산 최적화에 활용 가능",
      },
    ],
    metrics: {
      memory_reuse: "Near-Free When Fused",
      throughput: "Pointwise / Epilogue Bound",
      occupancy: "High or Piggybacked",
    },
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

  cudaCode: `// AICF: Fused ReLU (Epilogue)
// Inside GEMM / Conv / BiasAdd epilogue:
float acc = ...;          // upstream result
acc = fmaxf(acc, 0.0f);   // ReLU gating
*output = acc;            // final writeback`,
};