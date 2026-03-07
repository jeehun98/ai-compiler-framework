// src/data/analysis/configs/index.js
import { addAnalysis } from './add/index.js';
import { gemmAnalysis } from './gemm/index.js';
import { softmaxAnalysis } from './softmax/index.js';
import { adamStepAnalysis } from './adam_step/index.js';
import { batchNormAnalysis } from './batchnorm/index.js';
import { crossEntropyAnalysis } from './cross_entropy/index.js';

export const allAnalysisConfigs = {
  add: addAnalysis,
  gemm: gemmAnalysis,
  softmax: softmaxAnalysis,
  adam_step: adamStepAnalysis,
  batchnorm: batchNormAnalysis,
  cross_entropy: crossEntropyAnalysis,
};