// src/data/analysis/configs/adam_step/f32.js
import metrics from '../../metrics/adam_step_f32.json';

export const adamStepF32Config = {
  id: "f32",
  name: "FP32 AdamStep",
  tag: "Baseline",
  description:
    "Single-precision fused Adam update kernel using FP32 parameter/state storage and FP32 computation throughout.",

  algorithm: {
    title: "Fused AdamStep (FP32)",
    logic:
      "각 thread가 하나의 parameter index를 담당하며, grad / m / v / param을 읽어 1st moment, 2nd moment, bias correction, parameter update를 한 번에 수행합니다. 모든 데이터와 연산이 FP32로 유지되는 baseline optimizer update 경로입니다.",
    strategy: [
      "각 thread가 parameter 1개를 독립적으로 갱신",
      "param / m / v / grad 모두 FP32",
      "moment update, bias correction, sqrt 연산을 FP32 수행",
      "fused pointwise optimizer update 구조",
      "수치적 기준선 역할의 baseline 경로"
    ]
  },

  blueprint: {
    mem_access: "32-bit scalar load/store",
    instruction: "fused scalar update + sqrtf",
    vector_width: 1,
    alignment_req: "4-byte",
    code_snippet:
      "float g = grad[i]; float m_t = m[i]; float v_t = v[i]; m_t = beta1 * m_t + (1.0f - beta1) * g; v_t = beta2 * v_t + (1.0f - beta2) * g * g; param[i] -= lr * (m_t / bc1) / (sqrtf(v_t / bc2) + eps);"
  },

  metrics,
  features: [
    "FP32 Parameter Storage",
    "FP32 Optimizer State",
    "FP32 Compute",
    "Baseline Reference"
  ],
  insights: [
    "FP16과 동일한 fused Adam update 구조를 사용하지만 더 큰 데이터 폭을 가집니다.",
    "SM throughput이 FP16 대비 낮게 나타나며, 이 차이가 latency 증가로 이어졌습니다.",
    "수치적 기준선으로는 적절하지만 성능 측면에서는 FP16 경로보다 불리했습니다."
  ]
};