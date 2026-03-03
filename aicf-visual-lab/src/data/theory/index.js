// src/data/theory/index.js
import { gemmTheory } from "./gemm.js";
import { biasTheory } from "./bias.js"; // biasTheory도 같은 스키마로 만들 것

export const theoryByOpId = {
  [gemmTheory.id]: gemmTheory,
  [biasTheory.id]: biasTheory,
};

export const theoryOpIds = Object.keys(theoryByOpId);