import { gemmData } from './gemm';
import { biasAddData } from './bias_add';
import { residualAddData } from './residual_add';
import { layerNormData } from './layer_norm';
import { softmaxData } from './softmax';
import { adamStepData } from './adam_step';
import { batchNormData } from './batch_norm';
import { reluData } from './relu';
import { gemmEpilogueData } from './gemm_epilogue';

// Deep Dive 데이터 Import
import { gemmDeepDive } from './deepdive/gemm';
import { biasAddDeepDive } from './deepdive/bias_add';
import { adamStepDeepDive } from './deepdive/adam_step';
import { batchNormDeepDive } from './deepdive/batchnorm';
import { layerNormDeepDive } from './deepdive/layer_norm';
import { reluDeepDive } from './deepdive/relu';
import { softmaxDeepDive } from './deepdive/softmax';
import { gemmEpilogueDeepDive } from './deepdive/gemm_epilogue';

export const allOpsData = {
  AdamStep: {
    ...adamStepData,
    ...adamStepDeepDive,
  },
  
  BatchNorm: {
    ...batchNormData,
    ...batchNormDeepDive
  },
  
  BiasAdd: {
    ...biasAddData,
    ...biasAddDeepDive,
  },

  GEMM: {
    ...gemmData,
    ...gemmDeepDive,
  },

  GEMM_Epilogue: {
    ...gemmEpilogueData,
    ...gemmEpilogueDeepDive,
  },
  
  LayerNorm: {
    ...layerNormData,
    ...layerNormDeepDive,
  },

  ReLU: {
    ...reluData,
    ...reluDeepDive,
  },

  ResidualAdd: residualAddData,
  
  Softmax: {
    ...softmaxData,
    ...softmaxDeepDive,
  },
  
};
