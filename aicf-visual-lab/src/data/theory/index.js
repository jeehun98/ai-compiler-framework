// src/data/theory/index.js
import { gemmTheory } from "./gemm.js";
import { stateMergeTheory } from "./state_merge.js";

export const theoryByOpId = {
  [gemmTheory.id]: gemmTheory,           // "GEMM"
  [stateMergeTheory.id]: stateMergeTheory, // "STATE_MERGE"
};

export const theoryOpIds = Object.keys(theoryByOpId);

// TheoryPage에서 초기 렌더링 시 사용할 기본값
export const DEFAULT_THEORY_OP = gemmTheory.id;