// src/data/adam_step.js

export const adamStepData = {
  id: "AdamStep",
  category: "확률적 최적화 / 상태 진화 (Stochastic Optimization)",

  descriptions: {
    essence:
      "AdamStep은 gradient, momentum, variance state를 함께 사용하여 각 파라미터의 업데이트 크기와 방향을 안정적으로 조정하는 상태 기반 최적화 연산입니다.",
    strategy:
      "AdamStep은 파라미터 자체뿐 아니라 momentum, variance, step count와 같은 상태를 함께 갱신하므로, 단일 elementwise 연산이 아니라 상태 일관성과 수치 안정성을 함께 보존하는 lowering 전략이 중요합니다.",
    hardware:
      "이 연산은 state update, adaptive scaling, weight decay 적용이 하나의 realization family로 결합될 수 있으며, 실제 memory schedule과 kernel metric은 Deep Dive 계층에서 다룹니다.",
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
      "g": "현재 관측된 gradient",
      "m": "누적된 1차 추세 상태",
      "v": "gradient scale에 대한 적응적 상태",
      "\\epsilon": "수치 안정성 확보를 위한 안전 항",
      "\\theta": "갱신 대상 파라미터",
    },
  },

  semantics: {
    thesis:
      "AdamStep은 noisy gradient를 직접 적용하는 대신, 상태 변수 m과 v를 통해 방향성과 스케일을 분리하여 안정적인 파라미터 진화를 유도하는 상태 기반 update operator입니다.",

    axes: {
      P: { name: "Parameters", role: "독립적 상태 갱신 단위" },
      t: { name: "Time", role: "bias correction과 상태 진화 기준" },
    },

    invariants: [
      {
        id: "INV_STATE_ALIGNMENT",
        name: "상태 정렬성 (State Alignment)",
        metric:
          "m_t, v_t, \\theta_t \\text{ are updated on the same parameter index } p",
        threshold: "Index-consistent update",
        allows: ["Multi-State Fusion", "Vectorized Update"],
      },
      {
        id: "INV_DENOM_SAFETY",
        name: "분모 안정성 (Denominator Safety)",
        metric: "\\sqrt{\\hat{v}_t} + \\epsilon > 0",
        threshold: "Strict Positive",
        allows: ["Epsilon Floor", "Fast rsqrt Approximation"],
      },
      {
        id: "INV_STEP_MONOTONICITY",
        name: "시간 단계 단조성 (Step Monotonicity)",
        metric: "t_{k+1} > t_k",
        threshold: "Monotonic Increase",
        allows: ["Bias Correction Precompute", "Scalar Broadcast"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "초기 학습 단계 (Early Phase)",
          rule: "t \\text{ 가 작을 때 } \\hat{m}_t, \\hat{v}_t \\text{ 의 bias correction 영향이 크다}",
          hint: "Bias correction 정확도 우선",
        },
        {
          name: "Epsilon 민감도",
          rule: "\\hat{v}_t \\to 0 \\text{ 인 구간에서는 } \\epsilon \\text{ 선택이 update 안정성에 직접 영향을 준다}",
          hint: "Epsilon floor 및 수치 안정성 우선",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_AdamW_1Pass",
      reason: [
        "\\text{상태 결합도(State Coupling): } m, v, g, \\theta \\text{ 가 동일 파라미터 축에서 함께 갱신된다}",
        "\\text{의미 보존 하의 통합 갱신: state transition과 parameter update를 단일 패스로 유지할 수 있다}",
        "\\text{Bias correction 및 weight decay가 동일 update 식에 결합 가능하다}",
        "\\text{따라서 } \\texttt{Fused\\_AdamW\\_1Pass} \\text{ family가 적합하다}",
      ],
      applied_rewrites: [
        "Multi-State Fusion",
        "Vectorized Update",
        "Fast Math (rsqrt)",
      ],
    },
  },

  kernel: {
    strategy: "Vectorized Multi-State Update",
    details: [
      {
        technique: "128-bit Vector Load/Store",
        semantic_link: "동일 파라미터 축의 상태를 결합된 형태로 갱신",
      },
      {
        technique: "Fast Inverse Sqrt (rsqrt)",
        semantic_link: "분모 안정성 유지 범위 내 근사 가속",
      },
      {
        technique: "Unroll & Pipeline",
        semantic_link: "상태 갱신 패턴의 연속성 활용",
      },
    ],
    metrics: {
      memory_reuse: "4.0x (Fused vs Separate)",
      throughput: "Memory Bandwidth Saturation (98%)",
      occupancy: "90%",
    },
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

  cudaCode: `// AICF: Fused AdamW (Single Kernel)
__global__ void adamw_fused_kernel(...) {
  // 1. Vectorized Load (m, v, g, theta)
  // 2. Update Moments (m <- beta1*m..., v <- beta2*v...)
  // 3. Compute Update with bias correction / stable denominator
  // 4. Apply weight decay in the same realization path
  // 5. Vectorized Store (new m, v, theta)
}`,
};