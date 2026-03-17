// src/data/theory/index.js
import { gemmTheory } from "./gemm.js";
import { stateMergeTheory } from "./state_merge.js";
import { softmaxTheory } from "./softmaxm.js";
import { layerNormTheory } from "./layer_norm.js";
import { attentionTheory } from "./attention.js";
import { activationTheory } from "./activation.js";
import { weightedReductionTheory } from "./weighted_reduction.js";

export const theoryByOpId = {
  [gemmTheory.id]: gemmTheory,           // "GEMM"
  [stateMergeTheory.id]: stateMergeTheory, // "STATE_MERGE"
  [softmaxTheory.id]: softmaxTheory,
  [layerNormTheory.id]: layerNormTheory,
  [attentionTheory.id]: attentionTheory,
  [activationTheory.id]: activationTheory,
  [weightedReductionTheory.id]: weightedReductionTheory,
};

export const theoryOpIds = Object.keys(theoryByOpId);

// TheoryPage에서 초기 렌더링 시 사용할 기본값
export const DEFAULT_THEORY_OP = gemmTheory.id;