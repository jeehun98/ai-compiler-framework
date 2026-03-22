// src/data/ops/index.js

import { gemmData } from "./gemm";
import { biasAddData } from "./bias_add";
import { residualAddData } from "./residual_add";
import { layerNormData } from "./layer_norm";
import { softmaxData } from "./softmax";
import { adamStepData } from "./adam_step";
import { batchNormData } from "./batch_norm";
import { reluData } from "./relu";

// Deep Dive 데이터 Import
import { gemmDeepDive } from "../deepdive/gemm";
import { biasAddDeepDive } from "../deepdive/bias_add";
import { adamStepDeepDive } from "../deepdive/adam_step";
import { batchNormDeepDive } from "../deepdive/batch_norm";
import { layerNormDeepDive } from "../deepdive/layer_norm";
import { reluDeepDive } from "../deepdive/relu";
import { softmaxDeepDive } from "../deepdive/softmax";

const opRegistry = {
  AdamStep: {
    ...adamStepData,
    ...adamStepDeepDive,
  },

  BatchNorm: {
    ...batchNormData,
    ...batchNormDeepDive,
  },

  BiasAdd: {
    ...biasAddData,
    ...biasAddDeepDive,
  },

  GEMM: {
    ...gemmData,
    ...gemmDeepDive,
  },

  LayerNorm: {
    ...layerNormData,
    ...layerNormDeepDive,
  },

  ReLU: {
    ...reluData,
    ...reluDeepDive,
  },

  ResidualAdd: {
    ...residualAddData,
  },

  Softmax: {
    ...softmaxData,
    ...softmaxDeepDive,
  },
};

// 개발 중 id 불일치 방지
for (const [key, op] of Object.entries(opRegistry)) {
  if (op?.id !== key) {
    console.warn(
      `[ops/index] registry key "${key}" does not match op.id "${op?.id}"`
    );
  }
}

export const allOpsData = opRegistry;
export const allOpIds = Object.keys(opRegistry);
export const allOpsList = allOpIds.map((id) => opRegistry[id]);

export const opsByCategory = allOpsList.reduce((acc, op) => {
  const category = op.category || "Uncategorized";
  if (!acc[category]) acc[category] = [];
  acc[category].push(op);
  return acc;
}, {});

export const opPropertySummary = Object.fromEntries(
  allOpIds.map((opId) => [
    opId,
    Object.entries(opRegistry[opId]?.propertyProfile ?? {})
      .map(([propertyId, property]) => ({
        id: propertyId,
        status: property?.status ?? "unknown",
        score: property?.score ?? null,
      }))
      .sort((a, b) => (b.score ?? -1) - (a.score ?? -1)),
  ])
);

export const opsByProperty = (() => {
  const grouped = allOpIds.reduce((acc, opId) => {
    const propertyProfile = opRegistry[opId]?.propertyProfile ?? {};

    Object.entries(propertyProfile).forEach(([propertyId, property]) => {
      if (!acc[propertyId]) acc[propertyId] = [];
      acc[propertyId].push({
        opId,
        status: property?.status ?? "unknown",
        score: property?.score ?? null,
      });
    });

    return acc;
  }, {});

  Object.keys(grouped).forEach((propertyId) => {
    grouped[propertyId].sort((a, b) => {
      const scoreDiff = (b.score ?? -1) - (a.score ?? -1);
      if (scoreDiff !== 0) return scoreDiff;
      return a.opId.localeCompare(b.opId);
    });
  });

  return grouped;
})();

export const opsWithDeepDive = allOpIds.filter((opId) => {
  const op = opRegistry[opId];
  return Boolean(op?.kernel_evolution || op?.evolution || op?.mechanism);
});

export const opFamilyTraits = Object.fromEntries(
  allOpIds.map((opId) => {
    const op = opRegistry[opId];
    return [
      opId,
      {
        normalizationFamily: op.normalizationFamily ?? null,
        gatingFamily: op.gatingFamily ?? null,
        pathMergeFamily: op.pathMergeFamily ?? null,
        broadcastShiftFamily: op.broadcastShiftFamily ?? null,
        linearProjectionFamily: op.linearProjectionFamily ?? null,
        competitionFamily: op.competitionFamily ?? null,
        stateUpdateFamily: op.stateUpdateFamily ?? null,
      },
    ];
  })
);

/**
 * New structure:
 * opsByInvariant[opId][invariantProfileKey] = { status, score, reason }
 *
 * This matches InvariantPage.jsx:
 *   Object.entries(opsByInvariant).map(([opId, invariantMap]) => invariantMap[profileKey])
 */
export const opsByInvariant = {
  AdamStep: {
    reduction_equivalence: {
      status: "medium",
      score: 0.72,
      reason:
        "Optimizer update contains reduction-like aggregated state usage, but it is not primarily a reduction-defined operator.",
    },
    representation_equivalence: {
      status: "strong",
      score: 0.9,
      reason:
        "State update logic can be preserved across fused or packed internal representations when update semantics remain unchanged.",
    },
    bounded_numeric_drift: {
      status: "strong",
      score: 0.92,
      reason:
        "Optimizer updates are highly sensitive to precision and accumulation drift, so bounded drift is a primary safety condition.",
    },
    decision_tolerance: {
      status: "limited",
      score: 0.44,
      reason:
        "AdamStep usually affects training trajectory rather than an immediate downstream decision boundary.",
    },
  },

  BatchNorm: {
    reduction_equivalence: {
      status: "strong",
      score: 0.95,
      reason:
        "Batch statistics are computed through partial reduction and merged global state, so reduction equivalence is core.",
    },
    normalization_preservation: {
      status: "strong",
      score: 0.96,
      reason:
        "The operator is defined by normalization semantics, so statistic and denominator meaning must be preserved exactly in transformed execution.",
    },
    tiled_execution_equivalence: {
      status: "medium",
      score: 0.78,
      reason:
        "Partitioned statistic computation is allowed when local batch contributions are recomposed correctly.",
    },
    bounded_numeric_drift: {
      status: "strong",
      score: 0.89,
      reason:
        "Variance/mean estimation can drift under low precision or reordered accumulation, so numeric budget control matters.",
    },
  },

  BiasAdd: {
    domain_pruning_preservation: {
      status: "medium",
      score: 0.63,
      reason:
        "Bias application itself is simple, but masked or broadcast-elided regions may be skipped only if output meaning stays unchanged.",
    },
    representation_equivalence: {
      status: "strong",
      score: 0.87,
      reason:
        "BiasAdd is often fused into surrounding operators, so alternate internal representation must preserve the same output meaning.",
    },
    tiled_execution_equivalence: {
      status: "medium",
      score: 0.68,
      reason:
        "Tilewise execution is safe when broadcasting and output write boundaries are handled correctly.",
    },
    bounded_numeric_drift: {
      status: "medium",
      score: 0.62,
      reason:
        "The operator is not highly reduction-sensitive, but precision changes can still shift output values.",
    },
  },

  GEMM: {
    reduction_equivalence: {
      status: "strong",
      score: 0.96,
      reason:
        "GEMM relies on accumulation across the K dimension, so partial sums and merged accumulation must preserve the same aggregate meaning.",
    },
    tiled_execution_equivalence: {
      status: "strong",
      score: 0.98,
      reason:
        "Tiled execution is a primary realization strategy for GEMM, and correctness depends on sound local accumulation plus global recomposition.",
    },
    representation_equivalence: {
      status: "strong",
      score: 0.89,
      reason:
        "Packed/vectorized paths, fused epilogues, and alternate accumulator forms are legal only if matrix product meaning is preserved.",
    },
    bounded_numeric_drift: {
      status: "strong",
      score: 0.9,
      reason:
        "Mixed precision and reordered accumulation can introduce drift, so GEMM needs a controlled numeric budget.",
    },
    decision_tolerance: {
      status: "conditional",
      score: 0.79,
      reason:
        "When GEMM feeds logits, scores, or gates, downstream decisions such as argmax, top-k, or sign become the practical acceptance boundary.",
    },
  },

  LayerNorm: {
    reduction_equivalence: {
      status: "strong",
      score: 0.91,
      reason:
        "Mean and variance computation depend on correct local statistic accumulation and merge.",
    },
    normalization_preservation: {
      status: "strong",
      score: 0.97,
      reason:
        "LayerNorm is directly defined by normalized output structure, so transformed execution must preserve normalization semantics.",
    },
    tiled_execution_equivalence: {
      status: "medium",
      score: 0.71,
      reason:
        "Partitioned statistic computation can be valid, but boundary handling must preserve the full-row normalization target.",
    },
    bounded_numeric_drift: {
      status: "strong",
      score: 0.94,
      reason:
        "Norm computation is sensitive to precision and cancellation, so drift must remain tightly controlled.",
    },
    decision_tolerance: {
      status: "limited",
      score: 0.48,
      reason:
        "LayerNorm mainly affects distribution and scaling rather than a direct discrete downstream choice.",
    },
  },

  ReLU: {
    domain_pruning_preservation: {
      status: "strong",
      score: 0.96,
      reason:
        "Inactive negative domain can be safely pruned only if sign and threshold semantics remain unchanged.",
    },
    representation_equivalence: {
      status: "medium",
      score: 0.66,
      reason:
        "ReLU can be fused or represented implicitly, but the activation boundary must still match the original meaning.",
    },
    bounded_numeric_drift: {
      status: "conditional",
      score: 0.58,
      reason:
        "Small numeric drift is acceptable only when it does not cross the activation boundary near zero.",
    },
    decision_tolerance: {
      status: "strong",
      score: 0.93,
      reason:
        "Sign and threshold crossings directly change activation masks, making downstream boundary stability the key acceptance rule.",
    },
  },

  ResidualAdd: {
    representation_equivalence: {
      status: "medium",
      score: 0.74,
      reason:
        "ResidualAdd is frequently fused into larger blocks, so alternate form is acceptable if elementwise sum meaning remains unchanged.",
    },
    tiled_execution_equivalence: {
      status: "medium",
      score: 0.64,
      reason:
        "Tilewise partition is straightforward, but output boundary handling must remain exact.",
    },
    bounded_numeric_drift: {
      status: "medium",
      score: 0.69,
      reason:
        "Precision changes affect additive output, though the operator is less sensitive than normalization-heavy paths.",
    },
  },

  Softmax: {
    reduction_equivalence: {
      status: "strong",
      score: 0.94,
      reason:
        "Softmax depends on reduction-like denominator construction and mergeable state across tiles or streaming execution.",
    },
    normalization_preservation: {
      status: "strong",
      score: 0.99,
      reason:
        "Softmax is the canonical normalization-sensitive operator, so sum-to-one and denominator meaning must survive transformation.",
    },
    tiled_execution_equivalence: {
      status: "conditional",
      score: 0.73,
      reason:
        "Blockwise realization is possible, but only with correct rescaling and denominator recomposition.",
    },
    bounded_numeric_drift: {
      status: "strong",
      score: 0.98,
      reason:
        "Exponentiation and normalization make Softmax especially sensitive to overflow, underflow, and rescaling drift.",
    },
    decision_tolerance: {
      status: "strong",
      score: 0.88,
      reason:
        "Softmax outputs often feed ranking- or selection-sensitive logic, so downstream confidence and candidate order matter.",
    },
  },
};

export const invariantSummaryByOp = Object.fromEntries(
  allOpIds.map((opId) => [
    opId,
    Object.entries(opsByInvariant[opId] ?? {})
      .map(([invariantId, profile]) => ({
        id: invariantId,
        status: profile?.status ?? "unknown",
        score: profile?.score ?? null,
        reason: profile?.reason ?? "",
      }))
      .sort((a, b) => (b.score ?? -1) - (a.score ?? -1)),
  ])
);