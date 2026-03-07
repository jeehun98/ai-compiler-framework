// src/data/analysis/configs/adam_step/f16.js
import metrics from '../../metrics/adam_step_f16.json';

export const adamStepF16Config = {
  id: "f16",
  name: "FP16 AdamStep",
  tag: "Fast",
  description:
    "Mixed-precision Adam update kernel using FP16 parameter/state storage with FP32 internal computation for stability and efficiency.",

  algorithm: {
    title: "Fused AdamStep (FP16 I/O, FP32 Compute)",
    logic:
      "각 thread가 하나의 parameter index를 담당하며, grad / m / v / param을 읽어 1st moment, 2nd moment, bias correction, parameter update를 한 번에 수행합니다. 저장은 FP16이지만 내부 계산은 float로 수행되어 수치 안정성과 성능을 동시에 노립니다.",
    strategy: [
      "각 thread가 parameter 1개를 독립적으로 갱신",
      "param / m / v / grad는 FP16 저장",
      "moment update, bias correction, sqrt 연산은 FP32 수행",
      "fused pointwise optimizer update 구조",
      "메모리 footprint 감소로 bandwidth 부담 완화"
    ]
  },

  blueprint: {
    mem_access: "16-bit load/store with FP32 internal compute",
    instruction: "__half2float + fused scalar update + sqrtf",
    vector_width: 1,
    alignment_req: "2-byte",
    code_snippet:
      "float g = __half2float(grad[i]); float m_t = __half2float(m[i]); float v_t = __half2float(v[i]); m_t = beta1 * m_t + (1.0f - beta1) * g; v_t = beta2 * v_t + (1.0f - beta2) * g * g; float p = __half2float(param[i]); p -= lr * (m_t / bc1) / (sqrtf(v_t / bc2) + eps);"
  },

  metrics,
  features: [
    "FP16 Parameter Storage",
    "FP16 Optimizer State",
    "FP32 Internal Compute",
    "Fused Pointwise Update"
  ],
  insights: [
    "FP32 대비 warp active와 SM throughput이 모두 더 높게 나타났습니다.",
    "FP16 저장 형식을 사용하지만 실제 update 수식은 float로 계산됩니다.",
    "실측 결과 기준 FP32 대비 약 1.68배 더 짧은 실행 시간을 보였습니다."
  ]
};