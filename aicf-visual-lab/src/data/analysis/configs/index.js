// src/data/analysis/configs/index.js
import { addAnalysis } from './add/index.js';
import { gemmAnalysis } from './gemm/index.js';

export const allAnalysisConfigs = {
  add: addAnalysis,
  gemm: gemmAnalysis,
};