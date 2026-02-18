import { gemmData } from './gemm';
import { biasAddData } from './bias_add';
import { residualAddData } from './residual_add';
import { layerNormData } from './layer_norm';
import { softmaxData } from './softmax';
import { adamStepData } from './adam_step';
import { batchNormData } from './batch_norm';
import { reluData } from './relu';

// Deep Dive 데이터 Import
import { gemmDeepDive } from './deepdive/gemm';
import { biasAddDeepDive } from './deepdive/bias_add';
// import { layerNormDeepDive } from './deepdive/layer_norm';

export const allOpsData = {
  AdamStep: adamStepData,
  
  BatchNorm: batchNormData,
  
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
    // ...layerNormDeepDive,
  },

  ReLU: reluData,

  ResidualAdd: residualAddData,
  
  Softmax: softmaxData,
  
};
