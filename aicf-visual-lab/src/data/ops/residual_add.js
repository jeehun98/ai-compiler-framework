// src/data/residual_add.js

export const residualAddData = {
  id: "ResidualAdd",
  category: "잔차 병합 / 경로 합류 (Residual Path Merge)",

  descriptions: {
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
      M: { name: "Samples", role: "독립적으로 병합되는 샘플/토큰 축" },
      N: { name: "Features", role: "동일 좌표계에서 더해지는 특징/채널 축" },
    },

    invariants: [
      {
        id: "INV_SHAPE_ALIGNMENT",
        name: "형상 정렬성 (Shape Alignment)",
        metric: "shape(R) = shape(X) = shape(Y)",
        threshold: "Exact elementwise merge legality",
        allows: ["Pointwise Fusion", "In-Place Accumulation"],
      },
      {
        id: "INV_IDENTITY_PRESERVATION",
        name: "정체 경로 보존성 (Identity Preservation)",
        metric: "\\frac{\\partial Y}{\\partial R} = 1",
        threshold: "Exact identity contribution",
        allows: ["Residual Path Merge", "Gradient Highway Preservation"],
      },
      {
        id: "INV_ADDITIVE_COMPOSITION",
        name: "가법 합성성 (Additive Composition)",
        metric: "Y_{i,j} = R_{i,j} + X_{i,j}",
        threshold: "Elementwise additive consistency",
        allows: ["Epilogue Accumulate", "Fused Add+Norm"],
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
  },

  lowering: {
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
};