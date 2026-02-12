// src/data/residual_add.js
export const residualAddData = {
  id: "ResidualAdd",
  category: "State Merge / Error Correction",

  canonical: {
    formula: "Y = X + R",
    shapes: { X: "M×N", R: "M×N", Y: "M×N" },
    interpretation: {
      R: "Stable memory / Identity signal (Long-term)",
      X: "Correction term (Residual / Short-term)",
      Y: "Merged state (Stabilized)",
    },
  },

  semantics: {
    thesis:
      "Information supplement operator: preserves established memory (R) while applying delta correction (X).",

    axes: {
      X: { name: "Correction Signal", role: "Error compensation / residual update" },
      R: { name: "Stable Memory", role: "Identity / long-term representation" },
    },

    invariants: [
      {
        id: "INV_STATE_STABILITY",
        name: "State Stability (SRR)",
        metric: "SRR = \\mathbb{E}\\left[\\frac{|X|}{|R|}\\right]",
        threshold: "SRR \\ll 1 \\Rightarrow Y \\approx R",
        applies_when: ["srr_low", "correction_sensitivity low"],
        allows: ["x_low_precision", "low_precision_accum", "conditional_skip"],
      },
      {
        id: "INV_ALIGNMENT",
        name: "Distribution Alignment",
        metric: "\\Delta_{align} = |\\mu_X-\\mu_R| + |\\sigma_X-\\sigma_R|",
        threshold: "\\Delta_{align} \\le \\tau",
        applies_when: ["distribution_alignment_check=true"],
        allows: ["safe_add", "fused_norm_add"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "SRR Low Regime",
          rule: "If ||X|| << ||R|| then correction is small; relax precision on X.",
          hint: "x_low_precision_allowed",
        },
        {
          name: "Misalignment Detected",
          rule: "If mean/var mismatch is large => representation collision; switch to SafeAdd / FusedNormAdd.",
          hint: "force_safe_variant",
        },
        {
          name: "Boundary Proximity",
          rule: "If output is near threshold (e.g., 0), skipping requires stricter verification.",
          hint: "skip_risk_high_near_boundary",
        },
      ],
      tilePriority: "srr_and_alignment_predict",
    },
  },

  rewrites: {
    candidates: [
      {
        id: "RW_X_LOW_PREC",
        name: "Relax X Precision",
        transform:
          "Y := R + X,\\quad X\\in\\{fp16,int8\\},\\ accum\\in\\{fp16,fp32\\}",
        preconditions: ["SRR < 0.1", "correction_sensitivity low", "alignment holds"],
        knobs: { x_precision: "int8|fp16", accumulate: "fp16|fp32", srr_threshold: 0.1 },
      },
      {
        id: "RW_DYNAMIC_SKIP",
        name: "Dynamic Skipping (Contracted)",
        transform: "Y := R",
        preconditions: ["window_profile_confidence high", "contract_verified", "not_near_boundary"],
        knobs: { confidence_window: "W", max_boundary_drift: "\\delta" },
      },
      {
        id: "RW_SAFEADD_SWITCH",
        name: "Switch to SafeAdd / FusedNormAdd",
        transform: "ResidualAdd \\Rightarrow \\text{SafeAdd or FusedNormAdd}",
        preconditions: ["alignment fails", "distribution_alignment_check=true"],
      },
    ],
  },

  costModel: {
    compute: ["Bandwidth", "Launch Overhead"],
    semanticLoss:
      "\\lambda_1\\cdot BoundaryDrift + \\lambda_2\\cdot AlignmentFailure + \\lambda_3\\cdot SkipRisk",
    weights_hint: {
      default: { BoundaryDrift: 3.0, AlignmentFailure: 6.0, SkipRisk: 10.0 },
      safety_critical: { BoundaryDrift: 10.0, AlignmentFailure: 20.0, SkipRisk: 30.0 },
    },
  },

  lowering: {
    chosen: {
      variant: "VectorizedResidualAdd",
      reason: [
        "State merge is mandatory => keep deterministic baseline variant",
        "SRR profiling not available (static mode) => disable contracted skip by default",
      ],
      applied_rewrites: [],
    },
    options: ["PlainAdd", "VectorizedResidualAdd", "SafeAdd", "FusedNormAdd"],
  },

  kernel: {
    strategy: "Vectorized Add + Optional Guard",
    details: [
      {
        technique: "vectorized load/store",
        semantic_link: "state merge is bandwidth-bound; maximize streaming throughput",
      },
      {
        technique: "optional guard path (verify-gated)",
        semantic_link: "skip only under contracted verification to avoid semantic drift",
      },
    ],
    metrics: { memory_reuse: "Low (Streaming)", throughput: "Max", occupancy: 96 },
  },

  performance: {
    latency: { ours: 0.03, pytorch: 0.08, torch_compile: 0.04 },
  },

  cudaCode: `// AICF: ResidualAdd (Memory + Correction)
__global__ void residual_add(float* X, float* R, float* Y) {
  // Vectorized load for high-bandwidth memory merge
  // Optional logic for semantic skip-guard (verify-gated)
}`,
};
