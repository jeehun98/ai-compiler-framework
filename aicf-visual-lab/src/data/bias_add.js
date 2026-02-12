// src/data/bias_add.js
export const biasAddData = {
  id: "BiasAdd",
  category: "State Offset / Boundary Shifter",

  canonical: {
    formula: "Y = X + b",
    shapes: { X: "M×N", b: "1×N (broadcast)", Y: "M×N" },
    interpretation: {
      X: "Pre-activation field (Signal)",
      b: "Energy offset (Boundary Shift)",
      Y: "Shifted field (Calibrated)",
    },
  },

  semantics: {
    thesis:
      "Energy offset operator that calibrates decision boundaries, ensuring semantic alignment before activation.",

    axes: {
      X: { name: "Signal Field", role: "Pre-decision activation space" },
      b: { name: "Boundary Offset", role: "Energy / threshold calibration" },
    },

    invariants: [
      {
        id: "INV_TOPO_RANK",
        name: "Topological / Rank Preservation",
        metric: "\\Delta_{rank}(Y, X)",
        threshold: "pairwise\\ order\\ violations \\le \\epsilon",
        applies_when: ["downstream=TopK", "downstream=Ranking", "origin_rigidity=low"],
        allows: ["bias_low_precision", "integer_folding", "epilogue_fusion"],
      },
      {
        id: "INV_THRESHOLD_SAFETY",
        name: "Threshold Safety (Origin Rigidity)",
        metric: "\\Delta_{bdry}(Y, X; b)",
        threshold: "boundary\\ drift \\le \\delta",
        applies_when: ["downstream=Sigmoid", "downstream=threshold_ops", "origin_rigidity=high"],
        allows: ["safe_add", "higher_precision_bias"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "Ranking / TopK",
          rule: "Shift-invariant objectives tolerate aggressive folding and low-precision bias.",
          hint: "origin_rigidity = low",
        },
        {
          name: "Thresholded Ops (Sigmoid / HardTanh / etc.)",
          rule: "Absolute position matters; preserve bias precision to prevent boundary drift.",
          hint: "origin_rigidity = high",
        },
        {
          name: "ReLU Dead-zone",
          rule: "If profiled guarantee holds: max(X)+b < 0 => output becomes constant zero.",
          hint: "deadzone_trunc_candidate",
        },
      ],
      tilePriority: "boundary_proximity_predict",
    },
  },

  rewrites: {
    candidates: [
      {
        id: "RW_BIAS_FOLD",
        name: "Epilogue Bias Folding",
        transform: "BiasAdd \\Rightarrow \\text{fold into GEMM epilogue}",
        preconditions: ["no external intermediate", "layout compatible", "fusion boundary allowed"],
      },
      {
        id: "RW_BIAS_QUANT",
        name: "Quantized Bias (Contracted)",
        transform: "b \\in \\{fp16, int8\\}\\ \\text{with calibrated scale}",
        preconditions: ["origin_rigidity=low", "rank contract holds"],
        knobs: { min_precision: "int8|fp16", epsilon_rank_violation: "\\epsilon" },
      },
      {
        id: "RW_DEADZONE_TRUNC",
        name: "Semantic Dead-zone Truncation (Window-profiled)",
        transform: "\\max(X)+b < 0 \\Rightarrow Y := 0",
        preconditions: ["downstream=ReLU", "window_profile_confidence high"],
        knobs: { confidence_window: "W", bound_margin: "m" },
      },
    ],
  },

  costModel: {
    compute: ["Bandwidth", "Launch Overhead"],
    semanticLoss: "\\lambda_1\\cdot BoundaryDrift + \\lambda_2\\cdot OrderViolation",
    weights_hint: {
      default: { BoundaryDrift: 5.0, OrderViolation: 2.0 },
      safety_critical: { BoundaryDrift: 20.0, OrderViolation: 5.0 },
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_Epilogue_BiasAdd",
      reason: [
        "Bias is calibration state; prefer epilogue folding to avoid extra memory round-trip",
        "downstream contract indicates origin_rigidity=low => allow aggressive folding/precision",
      ],
      applied_rewrites: ["RW_BIAS_FOLD"],
    },
    options: ["PlainAdd", "VectorizedAdd", "Fused_Epilogue_BiasAdd", "SafeAdd"],
  },

  kernel: {
    strategy: "Vectorized Broadcast Add / Epilogue Fusion",
    details: [
      {
        technique: "vectorized load/store",
        semantic_link: "calibration merge is bandwidth-bound; maximize streaming efficiency",
      },
      {
        technique: "epilogue fusion",
        semantic_link: "complete semantic unit without materializing intermediate tensor",
      },
    ],
    metrics: { memory_reuse: "N/A (Streaming)", throughput: "High", occupancy: 98 },
  },

  performance: {
    latency: { ours: 0.02, pytorch: 0.06, torch_compile: 0.03 },
  },

  cudaCode: `// AICF: BiasAdd (prefer epilogue fold)
__global__ void fused_bias_add_kernel(...) {
  // Vectorized load and immediate fusion
  // avoiding extra memory round-trip
}`,
};
