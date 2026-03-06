// src/data/analysis/configs/index.js
import { addAnalysis } from './add/index.js';
import { gemmAnalysis } from './gemm/index.js';
import { softmaxAnalysis } from './softmax/index.js';

export const allAnalysisConfigs = {
  add: addAnalysis,
  gemm: gemmAnalysis,
  softmax: softmaxAnalysis,
};