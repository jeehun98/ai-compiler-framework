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
// import { biasAddDeepDive } from './deepdive/bias_add'; // (필요 시 주석 해제)
// import { layerNormDeepDive } from './deepdive/layer_norm'; // (필요 시 주석 해제)

export const allOpsData = {
  GEMM: {
    ...gemmData,
    ...gemmDeepDive // ✅ 여기서 병합! (kernel_evolution 등이 GEMM 객체 안에 들어감)
  },
  // 아직 Deep Dive 데이터가 없는 경우는 기존 데이터만 사용
  BiasAdd: {
    ...biasAddData,
    // ...biasAddDeepDive 
  },
  LayerNorm : {
    ...layerNormData,
    // ...layerNormDeepDive
  },
  
  ResidualAdd: residualAddData,
  Softmax : softmaxData,
  AdamStep : adamStepData, // 키 이름 통일 (S 대문자 권장)
  BatchNorm : batchNormData,
  ReLU : reluData,         // 키 이름 통일
};