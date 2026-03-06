// src/data/analysis/configs/gemm/f16.js
import metrics from '../../metrics/gemm_f16.json';

export const gemmWmmaF16Config = {
  id: "f16",
  name: "FP16 WMMA GEMM",
  tag: "Tensor Core",
  description:
    "WMMA-based FP16 GEMM using 16x16x16 tensor core fragments with FP32 accumulation and shared-memory staging.",

  algorithm: {
    title: "WMMA Tile GEMM (16x16x16)",
    logic:
      "각 block은 16x16 출력 타일 하나를 담당하며, A/B 타일을 shared memory에 pack한 뒤 WMMA fragment로 로드하여 tensor core mma_sync를 수행합니다. 누산은 float accumulator에서 진행되고 마지막에 __half로 저장됩니다.",
    strategy: [
      "출력 C의 16x16 tile 단위 계산",
      "A/B를 shared memory에 16x16 형태로 패킹",
      "wmma::load_matrix_sync + wmma::mma_sync 사용",
      "FP16 입력, FP32 accumulator, FP16 output",
      "K 차원을 16-step으로 순회"
    ]
  },

  blueprint: {
    mem_access: "16-bit tiled load + shared memory staging",
    instruction: "wmma::mma_sync (Tensor Core)",
    vector_width: 16,
    alignment_req: "16x16 fragment-friendly layout",
    code_snippet:
      "wmma::mma_sync(acc, a_frag, b_frag, acc); // FP16 WMMA GEMM with FP32 accumulation"
  },

  metrics: metrics,
  features: [
    "Tensor Core",
    "WMMA Fragment",
    "FP16 Input",
    "FP32 Accumulation",
    "Shared Memory Packing"
  ],
  insights: [
    "실행 시간은 F32 naive 대비 크게 짧지만 warp active 비율은 낮게 나타날 수 있습니다.",
    "낮은 occupancy처럼 보여도 tensor core 활용으로 인해 실제 처리량과 latency는 더 우수할 수 있습니다.",
    "shared memory packing 및 fragment load/store 구조가 성능에 직접적인 영향을 줍니다."
  ]
};