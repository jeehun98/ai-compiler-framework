// src/data/bias_add.js

export const biasAddData = {
  id: "BiasAdd",
  category: "상태 보정 / 경계 이동 (State Calibration)",

  descriptions: {
    essence:
      "BiasAdd는 각 특징 채널에 대해 상수 bias를 더하여 활성값의 기준점을 이동시키는 채널별 affine shift 연산입니다. 값의 상대적 차이는 유지한 채, 결정 경계와 후행 비선형 연산의 작동 위치를 조정합니다.",
    strategy:
      "BiasAdd는 독립적인 채널 축에 대해 동일한 broadcast add를 수행하므로, 단독 커널보다 선행 연산의 epilogue 또는 후행 pointwise 연산과 결합된 lowering이 자연스럽습니다. 핵심은 별도 메모리 왕복 없이 의미를 그대로 유지하는 것입니다.",
    realization:
      "주로 epilogue fusion 또는 pointwise fusion family로 연결되며, standalone kernel보다 fused realization이 더 자연스럽습니다. register injection과 bandwidth behavior의 상세 메커니즘은 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: "Y_{i,j} = X_{i,j} + b_j",
    shapes: {
      X: "M x N",
      b: "1 x N",
      Y: "M x N",
    },
    interpretation: {
      M: "샘플 축 또는 행 단위 처리 축",
      N: "채널/특징 축 (bias가 정의되는 축)",
      b: "채널별 기준점 이동값",
      "Y_{i,j}": "bias가 반영된 출력 활성값",
    },
  },

  semantics: {
    thesis:
      "BiasAdd는 각 채널에 대해 일정한 translation을 적용하는 affine-shift operator이며, 상대적 차이 구조를 유지하면서 후행 activation, normalization, residual merge의 기준점을 조정하는 역할을 합니다.",

    axes: {
      M: { name: "Samples", role: "독립적 데이터 행 또는 샘플 처리 축" },
      N: { name: "Features", role: "bias가 broadcast되는 채널/특징 축" },
    },

    invariants: [
      {
        id: "INV_CHANNEL_TRANSLATION",
        name: "채널별 평행 이동 (Channel-wise Translation)",
        metric: "Y_{i,j} - X_{i,j} = b_j",
        threshold: "Per-channel constant shift",
        allows: ["Broadcast Fusion", "Epilogue Injection"],
      },
      {
        id: "INV_RELATIVE_DIFFERENCE_PRESERVATION",
        name: "상대 차이 보존 (Relative Difference Preservation)",
        metric: "(Y_{i_1,j} - Y_{i_2,j}) = (X_{i_1,j} - X_{i_2,j})",
        threshold: "Difference-preserving shift",
        allows: ["Vectorized Add", "Streaming Realization"],
      },
      {
        id: "INV_BROADCAST_AXIS_CONSISTENCY",
        name: "브로드캐스트 축 일관성 (Broadcast Axis Consistency)",
        metric: "b_j \\text{ is shared across all } i \\text{ for fixed } j",
        threshold: "Axis-consistent broadcast",
        allows: ["Bias Cache Reuse", "Pointwise Fusion"],
      },
    ],

    downstreamConstraints: [
      {
        name: "ReLU / GELU Activation Boundary",
        rule:
          "BiasAdd \\text{ 는 activation 이전 기준점을 이동시키므로, 후행 비선형 함수의 활성 영역 분포를 직접 바꾼다}",
        hint: "Activation epilogue와 결합 우선",
      },
      {
        name: "LayerNorm / Mean-Centering",
        rule:
          "\\text{후행 연산이 평균 중심화(mean-centering)를 포함하면 일부 bias 효과는 정규화 단계에서 상쇄되거나 흡수될 수 있다}",
        hint: "Normalization-aware fusion 검토",
      },
      {
        name: "Residual Merge / Add Chain",
        rule:
          "\\text{후행 residual add와 연속된 pointwise chain에서는 standalone BiasAdd보다 fused pointwise realization이 유리하다}",
        hint: "Pointwise chain fusion",
      },
    ],
  },

  lowering: {
    chosen: {
      variant: "Fused_Epilogue_BiasAdd",
      summary:
        "BiasAdd는 채널 축에 대한 broadcast affine shift이므로, 선행 GEMM/Conv의 epilogue나 후행 pointwise chain 안에 직접 결합하는 realization이 자연스럽습니다.",
      reason: [
        "\\text{채널 broadcast 구조: } b_j \\text{ 는 출력 채널 축에만 의존하므로 GEMM/Conv 결과의 epilogue에서 직접 주입 가능하다}",
        "\\text{의미 보존 하의 통합: bias shift는 pointwise affine shift이므로 선행 연산 결과를 다시 읽지 않고 결합된 realization이 가능하다}",
        "\\text{중간 버퍼 회피: standalone BiasAdd를 제거하면 추가 load/store를 줄일 수 있다}",
      ],
      applied_rewrites: [
        "Epilogue Fusion",
        "Broadcast Injection",
        "Vectorized Pointwise Update",
      ],
    },
  },

  realizationSnapshot: {
    family: "Epilogue / Pointwise Broadcast Fusion",
    highlights: [
      "Epilogue register injection",
      "Broadcast bias reuse on feature axis",
      "Vectorized pointwise update",
      "Near-free when fused",
    ],
  },

  costModel: {
    semanticLoss:
      "\\mathcal{C}_{bias} = w_{axis} \\cdot \\Delta_{broadcast} + w_{fuse} \\cdot \\Delta_{fusion} + w_{num} \\cdot \\Delta_{numeric}",
    weights_hint: {
      default: {
        axis: 45.0,
        fuse: 35.0,
        numeric: 20.0,
      },
    },
    metrics: {
      broadcast_consistency: "High",
      epilogue_affinity: "Strong",
      numeric_risk: "Low",
    },
  },

  performance: {
    latency: {
      pytorch: 0.06,
      torch_compile: 0.03,
      ours: 0.01,
    },
  },
};