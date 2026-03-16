export const gemmData = {
  id: "GEMM",
  category: "선형 변환 / 특징 투영 (Linear Projection)",

  descriptions: {
    essence:
      "GEMM은 입력 표현을 새로운 특징 공간으로 투영하고, 샘플 축과 출력 채널 축 사이의 선형 결합을 생성하는 핵심 선형 연산입니다. 대부분의 딥러닝 블록에서 projection, mixing, scoring의 기본 단위를 이룹니다.",
    strategy:
      "GEMM은 K축 reduction과 출력 타일 축적을 중심으로 이루어지는 연산이므로, standalone matmul 자체뿐 아니라 bias, activation, residual add 같은 후행 pointwise 연산을 epilogue에 결합하는 lowering이 자연스럽습니다. 핵심은 중간 출력을 별도 메모리에 기록하지 않고 의미를 유지한 채 realization을 결합하는 것입니다.",
    hardware:
      "이 연산은 보통 Tensor Core / tiled reduction family로 연결되며, 실제 shared-memory tiling, register accumulation, pipeline staging 같은 구현 세부는 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: "C_{i,j} = \\alpha \\sum_{k=1}^{K} A_{i,k} B_{k,j} + \\beta C_{i,j}",
    shapes: {
      A: "M x K",
      B: "K x N",
      C: "M x N",
    },
    interpretation: {
      M: "출력 행을 형성하는 샘플/토큰 축",
      K: "누적(reduction)이 수행되는 내부 특징 축",
      N: "출력 채널 또는 투영 대상 특징 축",
      "A_{i,k}": "샘플 i의 k번째 입력 성분",
      "B_{k,j}": "입력 축 k를 출력 축 j로 사상하는 가중치",
      "C_{i,j}": "샘플 i의 출력 채널 j에 대한 투영 결과",
    },
  },

  semantics: {
    thesis:
      "GEMM은 K축을 따라 누적된 선형 결합을 통해 입력 표현을 새로운 출력 공간으로 사상하는 reduction-based projection operator이며, 후행 affine/activation 연산과 결합되기 쉬운 강한 epilogue 친화성을 가집니다.",

    axes: {
      M: { name: "Samples", role: "독립적 출력 행을 형성하는 축" },
      K: { name: "Reduction Axis", role: "누적 곱셈-덧셈이 이루어지는 내부 축" },
      N: { name: "Output Features", role: "출력 채널 또는 투영 목적 축" },
    },

    invariants: [
      {
        id: "INV_REDUCTION_EQUIVALENCE",
        name: "Reduction 동치성 (Reduction Equivalence)",
        metric:
          "C_{i,j} = \\sum_k A_{i,k}B_{k,j} \\text{ with equivalent accumulation over } K",
        threshold: "Equivalent reduction result",
        allows: ["Tiled Reduction", "Split-K", "TensorCore Accumulation"],
      },
      {
        id: "INV_OUTPUT_TILE_INDEPENDENCE",
        name: "출력 타일 독립성 (Output Tile Independence)",
        metric:
          "C_{i,j} \\text{ tiles are independently materializable before final writeback}",
        threshold: "Tile-local accumulation validity",
        allows: ["Block Tiling", "Warp Tiling", "Register Accumulation"],
      },
      {
        id: "INV_EPILOGUE_AFFINITY",
        name: "에필로그 결합성 (Epilogue Affinity)",
        metric:
          "\\tilde{C}_{i,j} = f(C_{i,j}) \\text{ where } f \\text{ is pointwise/affine on output elements}",
        threshold: "Output-local transform",
        allows: ["Bias Fusion", "Activation Fusion", "Residual Add Fusion"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "Bias / Activation Epilogue",
          rule:
            "\\text{후행 연산이 output-local pointwise 형태이면 } C_{i,j} \\text{ writeback 이전에 epilogue로 결합 가능하다}",
          hint: "Epilogue fusion 우선",
        },
        {
          name: "Softmax / Attention Score Use",
          rule:
            "\\text{출력이 softmax 입력으로 직접 사용되면 row-wise ordering과 numeric range가 후행 안정성에 큰 영향을 준다}",
          hint: "Numeric-stable GEMM epilogue 및 scaling 고려",
        },
        {
          name: "LayerNorm / Mean-Centering",
          rule:
            "\\text{후행 정규화가 출력 분포를 다시 조정하더라도 GEMM 자체의 reduction semantics와 output layout은 유지되어야 한다}",
          hint: "Normalization-aware lowering 검토",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "TensorCore_GEMM_EpilogueFused",
      reason: [
        "\\text{K축 reduction 구조: } A_{i,k}B_{k,j} \\text{ 누적은 tiled matmul realization으로 자연스럽게 분해 가능하다}",
        "\\text{출력 타일 독립성: } C \\text{ 의 부분 타일을 register/shared-memory에서 축적한 뒤 최종 writeback 할 수 있다}",
        "\\text{에필로그 결합성: bias/activation/residual과 같은 output-local transform은 matmul 결과 writeback 직전에 결합 가능하다}",
        "\\text{따라서 } \\texttt{TensorCore\\_GEMM\\_EpilogueFused} \\text{ family가 적합하다}",
      ],
      applied_rewrites: [
        "TensorCore Tiled Lowering",
        "Register Accumulation",
        "Epilogue Fusion",
      ],
    },
  },

  kernel: {
    strategy: "Hierarchical Tiled Reduction",
    details: [
      {
        technique: "Shared Memory Tiling",
        semantic_link: "K축 reduction에 필요한 A/B 조각 재사용",
      },
      {
        technique: "Register Accumulation",
        semantic_link: "출력 타일을 writeback 전까지 로컬하게 유지",
      },
      {
        technique: "Warp / TensorCore MMA",
        semantic_link: "작은 타일 단위의 선형 결합을 고처리량으로 실현",
      },
      {
        technique: "Epilogue Fusion",
        semantic_link: "출력-local 후행 연산을 별도 메모리 왕복 없이 결합",
      },
    ],
    metrics: {
      memory_reuse: "High (Tiled K-axis Reuse)",
      throughput: "TensorCore-Dominant",
      occupancy: "High",
    },
  },

  costModel: {
    semanticLoss:
      "\\mathcal{C}_{gemm} = w_{red} \\cdot \\Delta_{reduction} + w_{epi} \\cdot \\Delta_{epilogue} + w_{num} \\cdot \\Delta_{numeric}",
    weights_hint: {
      default: {
        reduction: 45.0,
        epilogue: 30.0,
        numeric: 25.0,
      },
    },
    metrics: {
      reduction_consistency: "High",
      epilogue_affinity: "Strong",
      numeric_sensitivity: "Moderate",
    },
    budget: {
      max_rel_error: "0.002",
      min_output_consistency: "0.999",
      epilogue_fusion_required: "Optional but preferred",
    },
  },

  performance: {
    latency: {
      pytorch: 0.92,
      torch_compile: 0.71,
      ours: 0.28,
    },
    config: {
      unit: "ms",
      device: "RTX 3060",
      dtype: "fp16",
      shape: "M=1024,K=4096,N=1024",
      batch: 256,
      measure: "cudaEvent avg over 100 iters",
    },
  },
};