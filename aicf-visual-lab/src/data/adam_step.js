// src/data/adam_step.js

export const adamStepData = {
  id: "AdamStep",
  category: "확률적 최적화 / 상태 진화 (Stochastic Optimization)",

  descriptions: {
    essence: "과거의 관성(Momentum)과 현재의 불확실성(Variance)을 동시에 고려하여, 최적의 파라미터 업데이트 경로를 탐색하는 항해사(Navigator)입니다.",
    strategy: "파라미터, 모멘텀, 분산 등 다수의 텐서 업데이트를 단일 커널로 묶어(Multi-Tensor Fusion), 메모리 대역폭을 한계까지 활용합니다.",
    hardware: "4번의 메모리 왕복을 1번의 커널로 통합(Fused Kernel)하여, GPU VRAM 대역폭의 한계치에 근접한 처리량을 달성합니다."
  },

  canonical: {
    formula: [
      "m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t",
      "v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2",
      "\\theta_{t+1} = \\theta_t - \\eta \\left( \\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t} + \\epsilon} + \\lambda \\theta_t \\right)"
    ].join("\\\\"),
    shapes: {
      "\\theta": "P (Parameters)",
      "g": "P (Gradients)",
      "m": "P (Momentum State)",
      "v": "P (Variance State)",
      "t": "Scalar (Time Step)"
    },
    interpretation: {
      "g": "현재의 관측 (Noisy Observation)",
      "m": "축적된 방향성 (Directional Inertia)",
      "v": "변화의 크기/불확실성 (Adaptive Scale)",
      "\\epsilon": "안전 댐퍼 (Safety Damper - 발산 방지)",
      "\\theta": "진화하는 지식 상태 (Evolving Knowledge)"
    },
  },

  semantics: {
    thesis: "순간적인 기울기(Gradient)의 잡음을 걸러내고, 과거의 경향성(Momentum)을 반영하여 파라미터를 안정적으로 진화시키는 시계열 제어 연산자",

    axes: {
      P: { name: "Parameters", role: "최적화 대상 (독립적 업데이트 단위)" },
      t: { name: "Time", role: "학습 진행도 (Bias Correction 기준)" },
    },

    invariants: [
      {
        id: "INV_TRUST_REGION",
        name: "신뢰 영역 보장 (Trust Region)",
        metric: "Update Magnitude Ratio",
        threshold: "급격한 파라미터 변화 억제",
        allows: ["Gradient Clipping", "Trust Region Policy"],
      },
      {
        id: "INV_DENOM_SAFETY",
        name: "분모 안정성 (Denominator Safety)",
        metric: "\\sqrt{v} + \\epsilon > 0",
        threshold: "Strict Positive",
        allows: ["Epsilon Lower Bound 강제"],
      },
      {
        id: "INV_STATE_CONSISTENCY",
        name: "상태 일관성 (State Consistency)",
        metric: "Monotonic Step Count",
        threshold: "t는 항상 증가",
        allows: ["Bias Correction 상수 미리 계산"],
      },
    ],


    sensitivity: {
      downstream: [
        {
          name: "초기 학습 단계 (Early Phase)",
          // 방법: 수식을 포함한 전체 문장을 LaTeX의 \text{} 기능을 활용해 작성
          rule: "t \\text{ 가 작을 때 } m, v \\text{ 의 0 편향 방지를 위한 Bias Correction 필수}",
          hint: "정밀 보정 모드 (Full Bias Correction)",
        },
        {
          name: "Epsilon 민감도",
          // 백슬래시 2개(\\) 사용 주의
          rule: "v \\to 0 \\text{ 일 때 } \\epsilon \\text{ 이 너무 작으면 업데이트 폭발 위험}",
          hint: "Epsilon Floor 정책 적용",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_AdamW_1Pass",
      reason: [
        "\\text{메모리 대역폭 병목(Memory Bound): } m, v, g, \\theta \\text{ 를 각각 읽고 쓰는 비용이 연산 비용을 압도함}",
        "\\text{커널 융합(Kernel Fusion): 4번의 메모리 접근을 1번의 통합 패스로 처리}",
        "\\text{레지스터 재사용: } g \\text{ 와 } m, v \\text{ 를 레지스터에서 바로 계산하여 L1/L2 캐시 오염 방지}",
        "\\text{AdamW 지원: Weight Decay를 별도 단계가 아닌 업데이트 수식에 통합}",
      ],
      applied_rewrites: ["Multi-Tensor Fusion", "Register Tiling", "Fast Math (rsqrt)"],
    },
  },

  kernel: {
    strategy: "Vectorized Multi-State Update",
    details: [
      { technique: "128-bit Vector Load/Store", semantic_link: "메모리 대역폭 100% 활용 (Float4)" },
      { technique: "Fast Inverse Sqrt (rsqrt)", semantic_link: "제어 이론상 허용 오차 내에서 나눗셈 가속" },
      { technique: "Unroll & Pipeline", semantic_link: "메모리 대기 시간(Latency) 은폐 (Compute Hiding)" },
    ],
    metrics: {
      memory_reuse: "4.0x (Fused vs Separate)",
      throughput: "Memory Bandwidth Saturation (98%)",
      occupancy: "90%"
    },
  },

  costModel: {
    semanticLoss: "\\mathcal{L}_{opt} = \\| \\theta_{true} - \\theta_{step} \\| + \\lambda \\cdot \\text{Stability}",
    weights_hint: {
      default: { convergence: 10.0, stability: 50.0 }
    },
    metrics: {
      update_stability: "High",
      nan_risk: "Controlled"
    }
  },

  performance: {
    latency: {
      pytorch: 2.50, // 개별 커널 실행 오버헤드 + 메모리 왕복
      torch_compile: 1.20,
      ours: 0.65 // 완벽하게 융합된 단일 커널
    }
  },
  
  cudaCode: `// AICF: Fused AdamW (Single Kernel)
__global__ void adamw_fused_kernel(...) {
  // 1. Vectorized Load (m, v, g, theta)
  // 2. Update Moments (m <- beta1*m..., v <- beta2*v...)
  // 3. Compute Update (Trust Region & Bias Correction)
  // 4. Apply Weight Decay (AdamW style)
  // 5. Vectorized Store (New m, v, theta)
  // All happening in registers without HBM round-trips
}`
};