export const adamStepData = {
  id: "AdamStep",
  category: "확률적 최적화 / 상태 진화 (Stochastic Optimization)",

  descriptions: {
    oneLine:
      "AdamStep은 gradient와 optimizer state(m, v)를 함께 갱신하는 parameter-local state transition operator입니다.",
    essence:
      "AdamStep은 gradient, momentum, variance state를 함께 사용하여 각 파라미터의 업데이트 크기와 방향을 안정적으로 조정하는 상태 기반 최적화 연산입니다.",
    strategy:
      "AdamStep은 파라미터뿐 아니라 momentum, variance, step count와 같은 상태를 함께 갱신하므로, 단순 elementwise 연산이 아니라 state consistency·numeric safety·parameter-local fusion 가능성을 함께 고려한 lowering 전략이 중요합니다.",
    realization:
      "주로 fused multi-state update family로 연결되며, state update·adaptive scaling·weight decay를 하나의 realization path로 결합할 수 있습니다. 상세 memory schedule과 kernel mechanism은 Deep Dive 계층에서 다룹니다.",
  },

  canonical: {
    formula: [
      "m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t",
      "v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2",
      "\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t}",
      "\\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}",
      "\\theta_{t+1} = \\theta_t - \\eta \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_t \\right)",
    ].join("\\\\"),
    shapes: {
      "\\theta": "P (Parameters)",
      "g": "P (Gradients)",
      "m": "P (Momentum State)",
      "v": "P (Variance State)",
      "t": "Scalar (Time Step)",
    },
    interpretation: {
      "\\theta": "갱신 대상 파라미터",
      g: "현재 관측된 gradient",
      m: "누적된 1차 추세 상태",
      v: "gradient scale에 대한 적응적 상태",
      t: "bias correction에 사용되는 step index",
      "\\epsilon": "수치 안정성 확보를 위한 안전 항",
    },
  },

  semantics: {
    thesis:
      "AdamStep은 noisy gradient를 직접 적용하는 대신, 상태 변수 m과 v를 통해 방향성과 스케일을 분리하여 안정적인 파라미터 진화를 유도하는 상태 기반 update operator입니다.",
    axes: {
      P: { name: "Parameters", role: "독립적 상태 갱신 단위" },
      t: { name: "Time", role: "bias correction과 상태 진화 기준" },
    },
  },

  propertyProfile: {
    local_accumulable: {
      status: "strong",
      score: 0.95,
      summary: "각 parameter index에서 m, v, g, theta를 로컬하게 읽고 갱신 후 writeback 가능",
      reason:
        "m, v, g, \\theta \\text{ are consumed and updated on the same parameter-local index}",
      allows: [
        "Multi-State Fusion",
        "Vectorized Update",
        "Single-Pass Writeback",
      ],
    },

    tile_composable: {
      status: "strong",
      score: 0.87,
      summary: "parameter 축을 block/tile 단위로 분할해도 의미 보존 가능",
      reason:
        "\\text{Parameter axis } P \\text{ can be partitioned into independent update tiles}",
      allows: ["Block Partitioning", "Vector Chunk Update", "Grid-Stride Lowering"],
    },

    order_rewritable: {
      status: "limited",
      score: 0.42,
      summary: "parameter axis 순서는 바꿀 수 있지만 time evolution 순서는 바꿀 수 없음",
      reason:
        "\\text{Reordering across } P \\text{ is allowed, but temporal state evolution over } t \\text{ is not commutative}",
      allows: ["Parameter Reindexing", "Independent Chunk Scheduling"],
    },

    schedule_invariant: {
      status: "limited",
      score: 0.46,
      summary: "intra-tile scheduling은 유연하지만 step별 state transition 순서는 유지되어야 함",
      reason:
        "\\text{Execution schedule may vary within a step, but } t_k \\rightarrow t_{k+1} \\text{ ordering must be preserved}",
      allows: ["Grid-Stride Schedule", "Warp/Block Local Rescheduling"],
    },

    representation_invariant: {
      status: "conditional",
      score: 0.68,
      summary: "벡터 폭, layout, register packing 변경은 가능하지만 state alignment가 깨지면 안 됨",
      reason:
        "\\text{Vector width / layout adaptation is allowed if } (m, v, g, \\theta) \\text{ remain index-aligned}",
      allows: ["128-bit Vectorization", "Packed Load/Store", "Contiguous Layout Preference"],
    },

    precision_relaxable: {
      status: "conditional",
      score: 0.58,
      summary: "fast rsqrt 등은 가능하지만 수치 손실 허용 범위가 제한적",
      reason:
        "\\text{Approximate math is possible if denominator stability and optimizer convergence remain bounded}",
      allows: ["Fast rsqrt Approximation", "Mixed Precision State Update"],
    },

    rematerializable: {
      status: "weak",
      score: 0.12,
      summary: "optimizer state 자체는 persistent state라 cheap recompute 대상이 아님",
      reason:
        "m \\text{ and } v \\text{ are persistent optimizer states, not ephemeral intermediates}",
      allows: ["Scalar Bias Factor Recompute"],
    },

    associative_merge: {
      status: "not_applicable",
      score: 0.05,
      summary: "자유로운 reduction merge 구조는 아님",
      reason:
        "\\text{Adam state transition is not a free associative merge over time}",
      allows: [],
    },

    domain_prunable: {
      status: "weak",
      score: 0.18,
      summary: "일반적으로 전체 parameter domain을 방문하지만 sparse/masked regime에서는 제한적 pruning 가능",
      reason:
        "\\text{General Adam update is dense, though sparse gradients may allow limited domain skipping}",
      allows: ["Sparse Update Shortcut"],
    },
  },

  opConstraints: [
    {
      id: "STATE_ALIGNMENT",
      name: "상태 정렬성 (State Alignment)",
      detail:
        "m, v, theta, gradient는 동일 parameter index 기준으로 정렬되어 갱신되어야 합니다.",
      metric:
        "m_t[p],\\ v_t[p],\\ g_t[p],\\ \\theta_t[p] \\text{ must refer to the same parameter index } p",
      consequence: ["State-Aligned Fusion", "Vectorized Update"],
    },
    {
      id: "DENOM_SAFETY",
      name: "분모 안정성 (Denominator Safety)",
      detail:
        "adaptive denominator는 항상 strict positive로 유지되어야 하며 epsilon floor가 이를 보조합니다.",
      metric: "\\sqrt{\\hat{v}_t} + \\epsilon > 0",
      consequence: ["Epsilon Floor", "Stable rsqrt Path"],
    },
    {
      id: "STEP_MONOTONICITY",
      name: "시간 단계 단조성 (Step Monotonicity)",
      detail:
        "bias correction과 state evolution은 step index 증가 순서를 보존해야 합니다.",
      metric: "t_{k+1} > t_k",
      consequence: ["Bias Correction Precompute", "Scalar Broadcast"],
    },
  ],

  downstreamConstraints: [
    {
      name: "초기 학습 단계 (Early Phase)",
      rule:
        "t \\text{ 가 작을 때 } \\hat{m}_t, \\hat{v}_t \\text{ 의 bias correction 영향이 크다}",
      hint: "Bias correction 정확도 우선",
    },
    {
      name: "Epsilon 민감도",
      rule:
        "\\hat{v}_t \\to 0 \\text{ 인 구간에서는 } \\epsilon \\text{ 선택이 update 안정성에 직접 영향을 준다}",
      hint: "Epsilon floor 및 수치 안정성 우선",
    },
    {
      name: "Weight Decay 결합",
      rule:
        "\\text{weight decay term이 동일 parameter-local update 식에 포함되면 state update와 단일 realization path로 결합 가능하다}",
      hint: "Fused AdamW 경로 우선",
    },
  ],

  lowering: {
    preferredFamily: "Vectorized Multi-State Update",
    candidates: [
      "Scalar Elementwise Update",
      "Vectorized Multi-State Update",
      "Fused_AdamW_1Pass",
    ],
    chosen: {
      variant: "Fused_AdamW_1Pass",
      summary:
        "AdamStep은 parameter-local state coupling이 강하고, m·v·g·theta가 동일 축에서 함께 소비되므로 fused multi-state update family가 가장 자연스럽습니다.",
      reason: [
        "\\text{상태 결합도(State Coupling): } m, v, g, \\theta \\text{ 가 동일 파라미터 축에서 함께 갱신된다}",
        "\\text{의미 보존 하의 통합 갱신: state transition과 parameter update를 단일 패스로 유지할 수 있다}",
        "\\text{Bias correction 및 weight decay가 동일 update 식에 결합 가능하다}",
      ],
      applied_rewrites: [
        "Multi-State Fusion",
        "Vectorized Update",
        "Fast Math (rsqrt)",
      ],
    },
  },

  realizationSnapshot: {
    family: "Vectorized Multi-State Update",
    highlights: [
      "128-bit vector load/store",
      "Fast inverse sqrt (rsqrt)",
      "State-aligned fused update",
      "Single-pass parameter writeback",
    ],
  },

  costModel: {
    semanticLoss:
      "\\mathcal{C}_{adam} = w_{state} \\cdot \\Delta_{state} + w_{num} \\cdot \\Delta_{numeric} + w_{step} \\cdot \\Delta_{schedule}",
    weights_hint: {
      default: {
        state: 40.0,
        numeric: 35.0,
        schedule: 15.0,
        convergence: 10.0,
      },
    },
    metrics: {
      state_consistency: "High",
      numeric_stability: "High",
      fused_update_affinity: "Strong",
    },
  },

  performance: {
    latency: {
      pytorch: 2.5,
      torch_compile: 1.2,
      ours: 0.65,
    },
  },

  stateUpdateFamily: {
    updateScope: "parameter-local",
    stateTensors: ["m", "v"],
    inputSignal: "gradient",
    temporalDependency: true,
    biasCorrection: true,
    weightDecayComposable: true,
    multiStateFusionFriendly: true,
  },

  cudaCode: `// AICF: Fused AdamW (Single Kernel)
__global__ void adamw_fused_kernel(...) {
  // 1. Vectorized Load (m, v, g, theta)
  // 2. Update Moments (m <- beta1*m..., v <- beta2*v...)
  // 3. Compute Update with bias correction / stable denominator
  // 4. Apply weight decay in the same realization path
  // 5. Vectorized Store (new m, v, theta)
}`,
};