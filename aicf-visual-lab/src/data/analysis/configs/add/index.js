// src/data/analysis/configs/add/index.js
import { f32Config } from './f32';
import { f16Config } from './f16';
import { f16x2Config } from './f16x2';

export const addAnalysis = {
  id: "add",
  label: "Element-wise Add",
  category: "Pointwise",
  // 최적화 순서대로 배열 구성 (UI 차트 정렬 기준)
  variants: [f16x2Config, f16Config, f32Config], 
  comparisonSummary: "Vectorization(half2)이 FP16 Scalar 대비 유의미한 명령어 발행 감소 및 처리 효율 향상을 보임."
};