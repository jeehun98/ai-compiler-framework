// src/data/adam_step.js
export const adamStepData = {
  id: "AdamStep",
  category: "Stateful Parameter Evolution / Stochastic Control Operator",

  canonical: {
    formula: [
      "m_t = \\beta_1 m_{t-1} + (1-\\beta_1) g_t",
      "v_t = \\beta_2 v_{t-1} + (1-\\beta_2) g_t^2",
      "\\hat{m}_t = \\frac{m_t}{1-\\beta_1^t},\\quad \\hat{v}_t = \\frac{v_t}{1-\\beta_2^t}",
      "\\theta_{t+1} = \\theta_t - \\eta\\,\\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}",
      "\\theta_{t+1}^{AdamW} = \\theta_t - \\eta\\left(\\frac{\\hat{m}_t}{\\sqrt{\\hat{v}_t}+\\epsilon}+\\lambda\\theta_t\\right)",
    ].join("\\\\"),
    shapes: {
      "\\theta": "P (same shape as g,m,v)",
      g: "P",
      m: "P",
      v: "P",
      t: "\\mathbb{N} (monotonic)",
    },
    interpretation: {
      g: "noisy observation (gradient)",
      m: "directional memory (EMA of gradient)",
      v: "scale/uncertainty memory (EMA of gradient^2)",
      "bias\\ correction": "initial-condition correction on EMA",
      "\\epsilon": "safety damper (denominator floor / noise amplifier limiter)",
      "\\lambda": "prior drift (decoupled weight decay; AdamW)",
      "\\theta": "state being evolved (parameters)",
    },
  },

  // 1) 의미론: '무엇을 보존해야 하는가'
  semantics: {
    thesis:
      "AdamStep is not a primitive optimizer kernel; it is a time-evolution operator over learning state, enforcing stability contracts while estimating control updates from noisy observations.",

    axes: {
      P: { name: "Parameters", role: "state dimension of \\theta, g, m, v" },
      t: { name: "Time Step", role: "monotonic global step; defines bias-correction dynamics" },
    },

    invariants: [
      {
        id: "INV_UPDATE_DIRECTION",
        name: "Update Direction Contract",
        metric: "\\cos(\\Delta\\theta,\\Delta\\theta')",
        threshold: "\\ge \\tau_{dir}",
        applies_when: ["semantic_equivalence_check=true", "window_verify=true"],
        allows: ["rsqrt_approx", "reduced_precision_moments", "bias_corr_fold (late)"],
      },
      {
        id: "INV_UPDATE_MAGNITUDE",
        name: "Step Magnitude Contract",
        metric: "\\|\\Delta\\theta'\\| / \\|\\Delta\\theta\\|",
        threshold: "\\in [1-\\delta,\\ 1+\\delta]",
        applies_when: ["semantic_equivalence_check=true", "window_verify=true"],
        allows: ["rsqrt_approx", "mixed_precision_update"],
      },
      {
        id: "INV_TRAJECTORY_WINDOW",
        name: "Convergence Behavior Contract (Window)",
        metric: "\\Delta\\mathcal{L}_{window},\\ \\mathrm{Var}(\\mathcal{L})_{window}",
        threshold: "drift \\le \\kappa",
        applies_when: ["training=true", "window_profile_confidence high"],
        allows: ["aggressive_rewrite_if_verified", "dynamic_policy_switch"],
      },
      {
        id: "INV_STATE_AUTHORITATIVE",
        name: "State Authoritativeness",
        metric: "\\{m,v,t\\} \\text{ consistency}",
        threshold: "no reset / monotonic t",
        applies_when: ["state_is_authoritative=true"],
        allows: ["fused_state_update_only", "no_recompute_state"],
      },
    ],

    stateMerge: {
      enabled: true,
      meaning:
        "(\\theta_t, m_{t-1}, v_{t-1}, t) \\mapsto (\\theta_{t+1}, m_t, v_t, t+1) is a single semantic unit (learning state transition).",
      params: {
        lr: "\\eta",
        beta1: "\\beta_1",
        beta2: "\\beta_2",
        eps: "\\epsilon",
        weight_decay: "\\lambda",
      },
      state_types: ["parameter_evolution", "memory_update", "control_update"],
    },

    attributes: {
      // Dynamics parameters
      lr: "\\eta",
      beta1: "\\beta_1",
      beta2: "\\beta_2",
      eps: "\\epsilon",
      weight_decay: "\\lambda (AdamW)",
      bias_correction: "true|false",
      step_t: "global_step_ref",

      // Stability / risk
      update_clip_threshold: "optional",
      denom_floor_policy: "strict|relaxed",
      nan_guard_policy: "abort|clamp|skip|report",

      // State semantics
      state_is_authoritative: true,
      state_reset_allowed: false,

      // Numeric mode
      param_dtype: "fp16|bf16|fp32",
      master_weight_fp32: "true|false",
      moment_dtype: "fp16|bf16|fp32",
      rsqrt_mode: "precise|approx",
      verify_mode: "true|false",
    },

    sensitivity: {
      downstream: [
        {
          name: "Epsilon / Denominator Safety",
          rule:
            "If \\min(\\sqrt{\\hat{v}}+\\epsilon) is small, updates can explode; treat \\epsilon as safety damper and enforce denom_floor_policy.",
          hint: "denom_min_guard",
        },
        {
          name: "LR Instability / NaN Risk",
          rule:
            "If update_norm spikes or nan_inf_rate rises, switch policies (clip / increase eps / disable approx) under contracted control rules.",
          hint: "dynamic_policy_switch",
        },
        {
          name: "Early-Step Bias Correction",
          rule:
            "Bias correction is essential at small t; folding/omission is forbidden until c1(t),c2(t) \\approx 1 and verified.",
          hint: "no_biascorr_fold_early",
        },
      ],
      tilePriority: "risk_score_predict (denom_min, update_norm, nan_inf_rate)",
    },
  },

  // 2) 허용 변형: '무엇을 바꿀 수 있는가'
  rewrites: {
    candidates: [
      {
        id: "RW_FUSED_STATE_UPDATE",
        name: "Fused State Update (1-pass)",
        transform:
          "(m,v,\\theta)\\ update\\ \\Rightarrow\\ single\\ kernel\\ pass\\ (semantic unit)",
        preconditions: ["state_is_authoritative=true", "no externally observed intermediate"],
        knobs: { vectorize: true, fuse_decay: true },
      },
      {
        id: "RW_BIASCORR_FOLD_LATE",
        name: "Bias Correction Folding (Late-phase, Contracted)",
        transform:
          "c_1(t)=\\frac{1}{1-\\beta_1^t},\\ c_2(t)=\\frac{1}{\\sqrt{1-\\beta_2^t}}\\ \\approx 1\\ \\Rightarrow\\ omit/approx",
        preconditions: ["t \\ge t_{warmup}", "c1,c2 within tol", "window_verify=true"],
        knobs: { t_warmup: "profiled", tol: "1e-3" },
      },
      {
        id: "RW_DENOM_RSQRT_APPROX",
        name: "Denominator Approximation (rsqrt LUT/NR)",
        transform:
          "\\frac{1}{\\sqrt{\\hat{v}}+\\epsilon}\\ \\Rightarrow\\ \\widetilde{\\mathrm{rsqrt}}(\\hat{v}+\\epsilon^2)",
        preconditions: ["direction/magnitude contracts hold", "denom_min above floor", "verify_mode=false OR verify_gated"],
        knobs: { method: "NR1|NR2|LUT", max_dir_drift: "\\tau_{dir}", max_mag_drift: "\\delta" },
      },
      {
        id: "RW_DECAY_FUSION",
        name: "Decoupled Weight Decay Fusion (AdamW)",
        transform: "\\lambda\\theta_t\\ \\Rightarrow\\ fused\\ into\\ update\\ pass",
        preconditions: ["weight_decay>0", "same pass update possible"],
        knobs: { fuse_decay: true },
      },
      {
        id: "RW_MIXED_PREC_STATE",
        name: "Mixed Precision State (Contracted)",
        transform: "m,v\\ in\\ fp16/bf16\\ with\\ fp32\\ accumulation\\ (guarded)",
        preconditions: ["window_verify=true", "nan_guard_policy active", "update contracts hold"],
        knobs: { moment_dtype: "bf16|fp16", accum: "fp32", guard: "denom_floor+clip" },
      },
      {
        id: "RW_NAN_GUARD",
        name: "NaN/Inf Guard Policy (Runtime)",
        transform: "detect NaN/Inf => clamp/skip/report under policy",
        preconditions: ["nan_guard_policy!=none"],
        knobs: { policy: "abort|clamp|skip|report", clamp_value: "profiled" },
      },
    ],
  },

  // 3) 비용함수: '무엇을 최소화하는가'
  costModel: {
    compute: ["bandwidth(state RW)", "rsqrt_cost", "elementwise_fma", "launch_overhead"],
    semanticLoss:
      "\\lambda_1\\cdot DirDrift + \\lambda_2\\cdot MagDrift + \\lambda_3\\cdot WindowConvergenceDrift + \\lambda_4\\cdot InstabilityRisk",
    weights_hint: {
      default: { DirDrift: 8.0, MagDrift: 6.0, WindowConvergenceDrift: 12.0, InstabilityRisk: 20.0 },
      safety_critical: { DirDrift: 20.0, MagDrift: 20.0, WindowConvergenceDrift: 35.0, InstabilityRisk: 50.0 },
    },
    semanticCompute:
      "Cost_{semantic} \\propto \\text{state traffic}(m,v) + \\text{risk guarding}(denom\\_floor, nan\\_guard)",
  },

  // 4) lowering 선택: '결국 어떤 커널을 택했는가'
  lowering: {
    chosen: {
      variant: "FusedAdamStep_StateRW_RsqrtApprox_Guarded",
      reason: [
        "AdamStep is a state transition => enforce fused 1-pass state update",
        "state traffic dominates => vectorize loads/stores, avoid extra intermediates",
        "rsqrt is hot => allow approx under direction/magnitude contracts",
        "enable runtime risk signals (denom_min, update_norm, nan_inf_rate) => dynamic policy switch",
        "AdamW drift term is prior => fuse weight decay into same pass",
      ],
      applied_rewrites: ["RW_FUSED_STATE_UPDATE", "RW_DENOM_RSQRT_APPROX", "RW_DECAY_FUSION", "RW_NAN_GUARD"],
    },
    options: [
      "AdamStep_PreciseFP32",
      "AdamStep_MixedPrecision",
      "FusedAdamStep_1Pass",
      "FusedAdamW_1Pass",
      "AdamStep_RsqrtApprox",
      "AdamStep_WithNaNGuard",
    ],
  },

  // 5) 물리 최적화: '어떻게 빨라졌는가'
  kernel: {
    strategy: "Vectorized 1-pass update over (g,m,v,theta) with guarded denom and optional decay",
    details: [
      { technique: "vectorized IO (128-bit/256-bit)", semantic_link: "state traffic dominates; maximize bandwidth efficiency" },
      { technique: "fused moment+param update", semantic_link: "learning state transition is a single semantic unit" },
      { technique: "rsqrt approx (NR/LUT)", semantic_link: "trade compute under Dir/Mag contracts; gate with denom_min" },
      { technique: "nan/inf guard + clipping", semantic_link: "control-system safety; prevent trajectory collapse" },
      { technique: "decoupled decay fused", semantic_link: "prior drift integrated without extra pass" },
    ],
    metrics: { memory_reuse: "Low (Streaming)", throughput: "Bandwidth-bound", occupancy: "—" },
  },

  performance: {
    latency: { ours: "—", pytorch: "—", torch_compile: "—" },
  },

  cudaCode: `// AICF: AdamStep (state transition, guarded, optionally AdamW)
__global__ void adam_step_fused(...) {
  // Inputs: g, theta, m, v, (optional) master_theta_fp32, step t, hyperparams
  // 1) m <- beta1*m + (1-beta1)*g
  // 2) v <- beta2*v + (1-beta2)*g*g
  // 3) bias correction (guarded; fold only late-phase)
  // 4) denom <- rsqrt(vhat) (+ eps safety damper), with denom floor policy
  // 5) update <- lr * mhat * denom (+ weight decay drift for AdamW)
  // 6) theta <- theta - update (with nan/inf guard / optional clipping)
}`,
};
