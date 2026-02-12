import { gemmData } from './gemm';
import { biasAddData } from './bias_add';
import { residualAddData } from './residual_add';
import { layerNormData } from './layer_norm';
import { softmaxData } from './softmax';
import { adamStepData } from './adam_step';
import { batchNormData } from './batch_norm';
import { reluData } from './relu';

import { gemmDeepDive } from './deepdive/gemm';

export const allOpsData = {
  GEMM: {
    ...gemmData,
    ...gemmDeepDive
  },
  BiasAdd: biasAddData,
  ResidualAdd: residualAddData,
  LayerNorm : layerNormData,
  Softmax : softmaxData,
  Adamstep : adamStepData,
  BatchNorm : batchNormData,
  Relu : reluData,
};
