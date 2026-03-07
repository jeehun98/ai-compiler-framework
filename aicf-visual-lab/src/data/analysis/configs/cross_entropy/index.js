// src/data/analysis/configs/cross_entropy/index.js
import { crossEntropyF32Config } from './f32.js';

export const crossEntropyAnalysis = {
  id: "cross_entropy",
  label: "Cross Entropy",
  category: "Loss",
  variants: [crossEntropyF32Config],
  comparisonSummary:
    "현재 분석 데이터는 FP32 cross-entropy forward의 핵심 reduction 커널에 대한 것입니다. 이 커널은 row-wise log-sum-exp 구조를 사용해 수치 안정적인 loss를 계산하며, row max / exp sum reduction 이후 최종 scalar loss를 atomicAdd로 누적합니다. 따라서 pure pointwise 커널보다는 reduction과 exp/log 연산 비용의 영향을 더 크게 받는 loss kernel로 해석하는 것이 적절합니다."
};