// src/data/analysis/configs/adam_step/index.js
import { adamStepF16Config } from './f16.js';
import { adamStepF32Config } from './f32.js';

export const adamStepAnalysis = {
  id: "adam_step",
  label: "Adam Step",
  category: "Optimizer",
  variants: [adamStepF16Config, adamStepF32Config],
  comparisonSummary:
    "두 커널은 모두 fused pointwise Adam update 구조를 사용하지만 데이터 타입 경로가 다릅니다. FP16 버전은 param, m, v, grad를 half로 저장하면서 내부 계산은 FP32로 수행해 수치 안정성을 유지했고, 더 낮은 메모리 footprint 덕분에 더 높은 warp active와 SM throughput을 기록하며 FP32 대비 약 1.68배 더 짧은 실행 시간을 달성했습니다."
};