// src/data/ops/gemm.js

export const gemmData = {
  id: "GEMM",
  category: "선형 변환 / 특징 투영 (Linear Projection)",

  descriptions: {
    oneLine:
      "GEMM은 K축 reduction을 통해 입력 표현을 새로운 출력 공간으로 투영하고, output-local epilogue와 강하게 결합될 수 있는 reduction-based projection operator입니다.",
    essence:
      "GEMM은 입력 표현을 새로운 특징 공간으로 투영하고, 샘플 축과 출력 채널 축 사이의 선형 결합을 생성하는 핵심 선형 연산입니다. 대부분의 딥러닝 블록에서 projection, mixing, scoring의 기본 단위를 이룹니다.",
    strategy:
      "GEMM은 K축 reduction과 출력 타일 축적을 중심으로 이루어지므로, standalone matmul뿐 아니라 bias, activation, residual add 같은 output-local 후행 연산을 epilogue에 결합하는 lowering이 자연스럽습니다. 핵심은 중간 출력을 별도 메모리에 기록하지 않고 의미를 유지한 채 realization을 결합하는 것입니다.",
    realization:
      "주로 Tensor Core 기반 tiled reduction family로 연결되며, 상세 kernel mechanism은 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula:
      "C_{i,j} = \\alpha \\sum_{k=1}^{K} A_{i,k} B_{k,j} + \\beta C_{i,j}",
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
      "GEMM은 K축을 따라 누적된 선형 결합을 통해 입력 표현을 새로운 출력 공간으로 사상하는 reduction-based projection operator이며, output-local 후행 연산과 결합되기 쉬운 강한 epilogue 친화성을 가집니다.",
    axes: {
      M: {
        name: "Samples",
        role: "독립적 출력 행을 형성하는 축",
        description:
          "출력 행은 샘플 또는 토큰 단위로 구성되며 row tile 분할의 기본 축이 됩니다.",
      },
      K: {
        name: "Reduction Axis",
        role: "누적 곱셈-덧셈이 이루어지는 내부 축",
        description:
          "K축은 partial accumulation과 merge가 반복되는 핵심 reduction domain입니다.",
      },
      N: {
        name: "Output Features",
        role: "출력 채널 또는 투영 목적 축",
        description:
          "출력 축은 epilogue 적용과 최종 writeback이 이루어지는 output-local domain입니다.",
      },
    },
  },

  propertyProfile: {
    associative_merge: {
      status: "strong",
      score: 0.95,
      summary:
        "K축 partial accumulation은 split-K, tile-level partial sum merge와 자연스럽게 결합됩니다.",
      reason:
        "\\text{Partial accumulations over reduction axis } K \\text{ are mergeable under equivalent GEMM reduction semantics}",
      allows: [
        "Split-K",
        "Partial Accumulation Merge",
        "TensorCore Accumulation",
      ],
    },

    local_accumulable: {
      status: "strong",
      score: 0.97,
      summary:
        "출력 타일은 register/shared-memory에서 local accumulation 후 최종 writeback 할 수 있습니다.",
      reason:
        "C_{tile} \\text{ can be accumulated locally before final materialization, preserving output-tile semantics}",
      allows: [
        "Register Accumulation",
        "Shared-Memory Accumulation",
        "Warp-Level MMA",
      ],
    },

    tile_composable: {
      status: "strong",
      score: 0.98,
      summary:
        "GEMM은 M/N/K 축을 기준으로 계층적 타일 분할이 매우 잘 맞는 대표 연산입니다.",
      reason:
        "\\text{GEMM semantics decompose naturally into hierarchical tiles over } (M,N,K)",
      allows: [
        "Block Tiling",
        "Warp Tiling",
        "CTA Tiling",
        "TensorCore Tile Lowering",
      ],
    },

    order_rewritable: {
      status: "conditional",
      score: 0.58,
      summary:
        "K축 accumulation 순서 재배치는 가능하지만 floating-point accumulation order에 따른 수치 차이는 존재합니다.",
      reason:
        "\\text{Reduction order over } K \\text{ may vary semantically, though floating-point accumulation order can affect numeric results}",
      allows: ["Reduction Reordering", "Split-K Scheduling"],
    },

    schedule_invariant: {
      status: "strong",
      score: 0.9,
      summary:
        "block/warp/pipeline schedule은 다양하게 바뀔 수 있으며 의미 보존 하에 고도 최적화가 가능합니다.",
      reason:
        "\\text{Execution schedule over hierarchical GEMM tiles is highly flexible while preserving projection semantics}",
      allows: [
        "Warp Specialization",
        "Pipeline Scheduling",
        "Async Copy Pipeline",
      ],
    },

    representation_invariant: {
      status: "strong",
      score: 0.91,
      summary:
        "layout, fragment packing, Tensor Core fragment representation 변경과 잘 맞습니다.",
      reason:
        "\\text{Data layout, fragment representation, and vectorized packing may vary if GEMM contraction semantics are preserved}",
      allows: [
        "TensorCore Fragment Mapping",
        "Vectorized Layout Transform",
        "Packed Load/Store",
      ],
    },

    rematerializable: {
      status: "weak",
      score: 0.11,
      summary:
        "출력 C는 재계산 비용이 커서 일반적으로 rematerialization 친화적이지 않습니다.",
      reason:
        "C \\text{ is expensive to recompute from } A,B \\text{, so GEMM outputs are not generally cheap rematerialization targets}",
      allows: [],
    },

    precision_relaxable: {
      status: "conditional",
      score: 0.69,
      summary:
        "fp16/tf32/tensorcore accumulation 등은 가능하지만 reduction 오차 budget을 관리해야 합니다.",
      reason:
        "\\text{Precision relaxation is possible if reduction consistency and output numeric budget remain acceptable}",
      allows: [
        "FP16 TensorCore Path",
        "TF32 Accumulation",
        "Mixed Precision GEMM",
      ],
    },

    domain_prunable: {
      status: "weak",
      score: 0.19,
      summary:
        "일반 dense GEMM은 전체 contraction domain을 사용하며 pruning은 sparsity 구조가 있을 때만 의미가 있습니다.",
      reason:
        "\\text{Dense GEMM generally requires the full contraction domain unless additional sparsity structure is available}",
      allows: ["Structured Sparsity Path"],
    },
  },

  opConstraints: [
    {
      id: "REDUCTION_EQUIVALENCE",
      name: "Reduction 동치성 (Reduction Equivalence)",
      detail:
        "K축을 따라 계산되는 누적 결과는 tiled/split 형태로 분해되더라도 동일 contraction 의미를 유지해야 합니다.",
      metric:
        "C_{i,j} = \\sum_k A_{i,k} B_{k,j} \\text{ with equivalent accumulation over } K",
      consequence: ["Tiled Reduction", "Split-K", "TensorCore Accumulation"],
    },
    {
      id: "OUTPUT_TILE_INDEPENDENCE",
      name: "출력 타일 독립성 (Output Tile Independence)",
      detail:
        "출력 타일은 로컬하게 축적 가능하며 최종 writeback 이전까지 독립적으로 유지될 수 있습니다.",
      metric:
        "C_{i,j} \\text{ tiles are independently materializable before final writeback}",
      consequence: ["Block Tiling", "Warp Tiling", "Register Accumulation"],
    },
    {
      id: "EPILOGUE_AFFINITY",
      name: "에필로그 결합성 (Epilogue Affinity)",
      detail:
        "bias, activation, residual add처럼 output-local transform은 GEMM 결과 writeback 직전에 결합될 수 있습니다.",
      metric:
        "\\tilde{C}_{i,j} = f(C_{i,j}) \\text{ where } f \\text{ is pointwise/affine on output elements}",
      consequence: ["Bias Fusion", "Activation Fusion", "Residual Add Fusion"],
    },
  ],

  downstreamConstraints: [
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

  lowering: {
    preferredFamily: "Hierarchical Tiled Reduction",
    candidates: [
      "SIMT_GEMM",
      "TensorCore_GEMM_EpilogueFused",
      "SplitK_GEMM",
    ],
    chosen: {
      variant: "TensorCore_GEMM_EpilogueFused",
      summary:
        "Reduction equivalence, output tile locality, epilogue affinity가 동시에 강하게 성립하므로 TensorCore epilogue-fused family가 자연스럽습니다.",
      reason: [
        "\\text{K축 reduction 구조는 tiled matmul realization으로 자연스럽게 분해 가능하다}",
        "\\text{출력 타일은 register/shared-memory에서 축적 후 최종 writeback 가능하다}",
        "\\text{Bias/activation/residual과 같은 output-local transform은 writeback 직전에 결합 가능하다}",
      ],
      applied_rewrites: [
        "TensorCore Tiled Lowering",
        "Register Accumulation",
        "Epilogue Fusion",
      ],
    },
  },

  realizationSnapshot: {
    family: "Hierarchical Tiled Reduction",
    highlights: [
      "Shared-memory tiling",
      "Register accumulation",
      "Warp / TensorCore MMA",
      "Epilogue fusion",
    ],
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

  linearProjectionFamily: {
    contraction: "A[M,K] x B[K,N] -> C[M,N]",
    reductionAxis: "K",
    outputTiling: true,
    epilogueFusion: true,
    tensorCoreFriendly: true,
    outputMaterialization: "preferred-after-epilogue",
  },
};