import { gemmTheory } from "./gemm";

// TODO: 다른 op 이론 파일 추가
// import { softmaxTheory } from "./softmax";
// import { layerNormTheory } from "./layernorm";

export const theoryByOpId = {
  GEMM: gemmTheory,
  // Softmax: softmaxTheory,
  // LayerNorm: layerNormTheory,
};

export const DEFAULT_THEORY_OP = "GEMM";