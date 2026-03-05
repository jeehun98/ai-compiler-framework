// src/data/analysis/configs/index.js
import { addAnalysis } from './add/index.js';

export const allAnalysisConfigs = {
  add: addAnalysis,
  // 추후 gemm: gemmAnalysis 등으로 확장 가능
};