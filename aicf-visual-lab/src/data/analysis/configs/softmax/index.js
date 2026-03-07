// src/data/analysis/configs/softmax/index.js
import { softmaxF16Config } from './f16.js';
import { softmaxF32Config } from './f32.js';

export const softmaxAnalysis = {
  id: "softmax",
  label: "Softmax",
  category: "Normalization",
  variants: [softmaxF16Config, softmaxF32Config],
  comparisonSummary:
  "두 softmax 커널은 모두 row-wise last-dimension reduction 구조를 사용하며 warp shuffle과 block reduction 패턴도 유사합니다. 차이는 주로 데이터 타입에 있으며, FP16 버전은 입력/출력 대역폭을 줄이면서 내부 reduction과 exp 계산은 FP32로 유지해 약 1.34배 더 짧은 실행 시간을 달성했습니다."
};