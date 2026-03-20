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

export const opsByInvariant = {
  semantic_consistency: [
    { opId: "gemm", status: "strong", score: 0.96 },
    { opId: "softmax", status: "strong", score: 0.94 },
    { opId: "layernorm", status: "medium", score: 0.82 },
  ],
  numeric_stability: [
    { opId: "softmax", status: "strong", score: 0.98 },
    { opId: "layernorm", status: "strong", score: 0.95 },
    { opId: "rmsnorm", status: "strong", score: 0.93 },
    { opId: "gemm", status: "conditional", score: 0.74 },
  ],
  structural_preservation: [
    { opId: "gemm", status: "strong", score: 0.95 },
    { opId: "matmul", status: "strong", score: 0.93 },
    { opId: "attention", status: "medium", score: 0.81 },
    { opId: "softmax", status: "conditional", score: 0.72 },
  ],
};