// src/data/ops/residual_add.js

export const residualAddData = {
  id: "ResidualAdd",
  category: "잔차 병합 / 경로 합류 (Residual Path Merge)",

  descriptions: {
    oneLine:
      "ResidualAdd는 identity path와 residual path를 같은 좌표계에서 병합하여 기존 표현을 보존하면서 상태를 점진적으로 갱신하는 same-shape path-merge operator입니다.",
    essence:
      "ResidualAdd는 기존 경로의 상태(identity path)와 새로 계산된 변화량(residual path)을 같은 좌표계에서 합쳐, 표현을 보존하면서도 점진적으로 갱신하는 경로 병합 연산입니다.",
    strategy:
      "ResidualAdd는 단순한 element-wise add처럼 보이지만 실제로는 두 실행 경로의 합류 지점이므로, standalone add보다 선행 연산의 epilogue 또는 후행 normalization과 결합된 lowering이 중요합니다. 핵심은 별도 중간 버퍼 없이 경로 병합 의미를 유지하는 것입니다.",
    realization:
      "주로 same-shape pointwise merge family로 연결되며, fused residual merge나 Add+Norm preparation path가 자연스럽습니다. in-place accumulate, epilogue path merge, fused add+norm의 상세 메커니즘은 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: "Y_{i,j} = R_{i,j} + X_{i,j}",
    shapes: {
      R: "M x N",
      X: "M x N",
      Y: "M x N",
    },
    interpretation: {
      M: "샘플/토큰 축",
      N: "특징/채널 축",
      R: "기존 상태를 전달하는 identity 경로",
      X: "새로 계산된 residual 변화량",
      "Y_{i,j}": "두 경로가 병합된 최종 상태",
    },
  },

  semantics: {
    thesis:
      "ResidualAdd는 기존 표현을 완전히 대체하지 않고 identity path와 residual path를 합쳐 상태를 진화시키는 path-merge operator입니다. 이 연산은 skip connection의 의미를 보존하며, 깊은 네트워크에서 정보 전달과 gradient flow를 안정화하는 구조적 역할을 가집니다.",
    axes: {
      M: {
        name: "Samples",
        role: "독립적으로 병합되는 샘플/토큰 축",
        description:
          "각 샘플 또는 토큰 row는 다른 row와 독립적으로 residual merge 됩니다.",
      },
      N: {
        name: "Features",
        role: "동일 좌표계에서 더해지는 특징/채널 축",
        description:
          "각 feature 위치는 identity 값과 residual 값을 같은 좌표에서 직접 병합합니다.",
      },
    },
  },

  propertyProfile: {
    associative_merge: {
      status: "medium",
      score: 0.62,
      summary:
        "elementwise additive composition 자체는 결합적이지만, 본질은 reduction merge가 아니라 same-shape path merge입니다.",
      reason:
        "\\text{Elementwise addition is associative, though ResidualAdd is primarily a path-merge operator rather than a reduction operator}",
      allows: [
        "Elementwise Merge Reassociation",
        "Path Merge Grouping",
      ],
    },

    local_accumulable: {
      status: "strong",
      score: 0.99,
      summary:
        "각 출력 원소는 동일 좌표의 identity/residual 값 두 개만 필요하므로 output-local merge가 매우 강합니다.",
      reason:
        "Y_{i,j} \\text{ depends only on } R_{i,j} \\text{ and } X_{i,j}, \\text{ making residual merge fully output-local}",
      allows: [
        "Pointwise Fusion",
        "In-Place Accumulation",
        "Producer-Epilogue Merge",
      ],
    },

    tile_composable: {
      status: "strong",
      score: 0.97,
      summary:
        "same-shape pointwise merge이므로 domain을 어떤 타일로 쪼개도 의미 보존이 쉽습니다.",
      reason:
        "\\text{Any tiling over } (M,N) \\text{ preserves semantics because each merged output element is independent}",
      allows: [
        "Block Tiling",
        "Vector Tile Add",
        "Elementwise Partitioning",
      ],
    },

    order_rewritable: {
      status: "strong",
      score: 0.94,
      summary:
        "원소 간 독립이라 처리 순서 재배치가 자유롭고 병렬화가 쉽습니다.",
      reason:
        "\\text{Evaluation order may vary freely because each residual merge is elementwise independent}",
      allows: [
        "Element Reordering",
        "Parallel Scheduling",
        "Vectorized Traversal",
      ],
    },

    schedule_invariant: {
      status: "strong",
      score: 0.92,
      summary:
        "standalone add, producer epilogue merge, add+norm 준비 경로 등 다양한 스케줄로 바뀔 수 있습니다.",
      reason:
        "\\text{Execution schedule is highly flexible because path merge is pointwise and can be fused with adjacent producers / consumers}",
      allows: [
        "Producer-Epilogue Merge",
        "Add+Norm Scheduling",
        "In-Place Merge Schedule",
      ],
    },

    representation_invariant: {
      status: "strong",
      score: 0.88,
      summary:
        "vectorized add, alias-aware in-place accumulate, fused writeback 등 다양한 표현으로 옮기기 쉽습니다.",
      reason:
        "\\text{Layout, vectorization, and in-place / fused representations preserve same-shape residual merge semantics naturally}",
      allows: [
        "Vectorized Add",
        "Alias-Aware In-Place Update",
        "Fused Writeback Merge",
      ],
    },

    rematerializable: {
      status: "medium",
      score: 0.57,
      summary:
        "identity path와 residual path 값이 있으면 merge 결과는 쉽게 재계산할 수 있습니다.",
      reason:
        "Y = R + X \\text{ is cheap to recompute once both path values are available}",
      allows: [
        "Cheap Recompute",
        "Intermediate Merge Elision",
      ],
    },

    precision_relaxable: {
      status: "conditional",
      score: 0.55,
      summary:
        "단순 add 자체는 안정적이지만 깊은 residual chain에서는 누적 오차에 주의가 필요합니다.",
      reason:
        "\\text{Addition is numerically simple, though repeated residual accumulation can still introduce drift under reduced precision}",
      allows: [
        "Mixed Precision Add",
        "Vectorized Accumulate",
      ],
    },

    domain_prunable: {
      status: "weak",
      score: 0.13,
      summary:
        "일반 residual add는 전체 same-shape domain을 방문하므로 pruning 여지는 작습니다.",
      reason:
        "\\text{ResidualAdd generally requires full elementwise participation over the aligned domain}",
      allows: [],
    },
  },

  opConstraints: [
    {
      id: "SHAPE_ALIGNMENT",
      name: "형상 정렬성 (Shape Alignment)",
      detail:
        "identity path와 residual path는 동일 shape과 좌표계를 가져야 하며, 그래야 elementwise merge가 합법적입니다.",
      metric: "shape(R) = shape(X) = shape(Y)",
      consequence: ["Pointwise Fusion", "In-Place Accumulation"],
    },
    {
      id: "IDENTITY_PRESERVATION",
      name: "정체 경로 보존성 (Identity Preservation)",
      detail:
        "ResidualAdd는 기존 경로를 제거하지 않고 그대로 전달하므로 identity contribution이 정확히 1로 유지되어야 합니다.",
      metric: "\\frac{\\partial Y}{\\partial R} = 1",
      consequence: ["Residual Path Merge", "Gradient Highway Preservation"],
    },
    {
      id: "ADDITIVE_COMPOSITION",
      name: "가법 합성성 (Additive Composition)",
      detail:
        "같은 좌표계에서 identity path와 residual path가 정확히 더해져야 하며, 이 성질이 epilogue accumulate와 Add+Norm fusion의 기반입니다.",
      metric: "Y_{i,j} = R_{i,j} + X_{i,j}",
      consequence: ["Epilogue Accumulate", "Fused Add+Norm"],
    },
  ],

  downstreamConstraints: [
    {
      name: "LayerNorm / RMSNorm Coupling",
      rule:
        "\\text{ResidualAdd 이후 바로 normalization이 오면 path merge와 row-wise statistics가 연속되므로 fused Add+Norm lowering이 유리하다}",
      hint: "Add+Norm fusion 우선 검토",
    },
    {
      name: "Transformer Residual Block",
      rule:
        "\\text{attention/MLP 출력이 identity path와 합쳐지는 경우 standalone add보다 block-level merge realization이 더 자연스럽다}",
      hint: "Residual block aware lowering",
    },
    {
      name: "In-Place Legality",
      rule:
        "\\text{identity buffer가 이후 독립적으로 재사용되지 않는다면 } Y \\leftarrow R + X \\text{ 형태의 in-place accumulate가 가능할 수 있다}",
      hint: "Alias / buffer reuse 검토",
    },
  ],

  lowering: {
    preferredFamily: "Same-Shape Pointwise Path Merge",
    candidates: [
      "Standalone_Pointwise_Add",
      "Fused_Epilogue_ResidualAdd",
      "Add_Norm_Preparation_Path",
    ],
    chosen: {
      variant: "Fused_Epilogue_ResidualAdd",
      summary:
        "ResidualAdd는 동일 shape의 identity path와 residual path를 output-local하게 병합하는 구조이므로, standalone add보다 fused residual merge나 Add+Norm 준비 경로가 더 자연스럽습니다.",
      reason: [
        "\\text{경로 병합 구조: } R \\text{ 과 } X \\text{ 는 동일 shape의 상태 텐서이므로 output-local merge가 가능하다}",
        "\\text{의미 보존 하의 통합: 선행 연산 결과 } X \\text{ 를 별도 버퍼에 기록하기 전에 identity path } R \\text{ 와 결합할 수 있다}",
        "\\text{중간 버퍼 제거: standalone residual add를 없애면 추가 load/store를 줄일 수 있다}",
      ],
      applied_rewrites: [
        "Residual Path Fusion",
        "In-Place Accumulation",
        "Add+Norm Ready Merge",
      ],
    },
  },

  realizationSnapshot: {
    family: "Same-Shape Pointwise Path Merge",
    highlights: [
      "Residual path fusion",
      "Vectorized same-shape add",
      "Optional in-place accumulation",
      "Add+Norm ready merge path",
    ],
  },

  costModel: {
    semanticLoss:
      "\\mathcal{C}_{res} = w_{id} \\cdot \\Delta_{identity} + w_{merge} \\cdot \\Delta_{merge} + w_{alias} \\cdot \\Delta_{buffer}",
    weights_hint: {
      default: {
        identity: 45.0,
        merge: 35.0,
        alias: 20.0,
      },
    },
    metrics: {
      identity_preservation: "High",
      merge_affinity: "Strong",
      in_place_potential: "Moderate-High",
    },
  },

  performance: {
    latency: {
      pytorch: 0.08,
      torch_compile: 0.04,
      ours: 0.0,
    },
  },

  pathMergeFamily: {
    mergeType: "elementwise-add",
    identityPath: true,
    residualPath: true,
    outputLocal: true,
    inPlacePossible: true,
    normFusionFriendly: true,
  },
};