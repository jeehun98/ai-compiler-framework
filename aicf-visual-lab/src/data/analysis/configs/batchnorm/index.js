// src/data/analysis/configs/batchnorm/index.js
import { batchNormF16AtomicConfig } from './f16_atomic.js';

export const batchNormAnalysis = {
  id: "batchnorm",
  label: "Batch Normalization",
  category: "Normalization",
  variants: [batchNormF16AtomicConfig],
  comparisonSummary:
    "현재 BatchNorm 분석은 end-to-end forward 전체가 아니라 hotspot 후보인 stats 커널에 집중합니다. bn_fwd_stats_f16_atomic는 FP16 입력을 읽어 channel-wise sum/sumsq를 FP32 atomicAdd로 누적하는 구조이며, 전체 파이프라인 중 먼저 분리 분석할 가치가 큰 단계입니다. 특히 매우 낮은 SM throughput은 연산량 부족보다 atomic contention 및 serialization 병목 가능성을 강하게 시사합니다."
};