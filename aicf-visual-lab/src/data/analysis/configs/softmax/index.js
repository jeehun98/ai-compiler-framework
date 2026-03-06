// src/data/analysis/configs/softmax/index.js
import { softmaxF16Config } from './f16.js';
import { softmaxF32Config } from './f32.js';

export const softmaxAnalysis = {
  id: "softmax",
  label: "Softmax",
  category: "Normalization",
  variants: [softmaxF16Config, softmaxF32Config],
  comparisonSummary:
    "FP16 Softmax는 FP32 Softmax와 거의 동일한 warp active를 유지하면서도 더 높은 SM throughput을 보였고, 결과적으로 약 1.34배 더 짧은 실행 시간을 달성했습니다. 이 결과는 softmax가 occupancy 차이보다 데이터 폭과 처리 효율의 영향을 더 크게 받는다는 점을 보여줍니다."
};