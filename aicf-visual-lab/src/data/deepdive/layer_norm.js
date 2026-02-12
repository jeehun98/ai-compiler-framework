// src/data/deepdive/layer_norm.js

export const layerNormDeepDive = {
  id: "LayerNorm",

  // KernelDeepDive.jsx가 쓰는 키
  kernel_evolution: [
    {
      version: "v1.0",
      tag: "Two-Pass (Mean/Var) Baseline",
      throughput: "—",
      description:
        "mean/var를 별도 패스로 계산한 뒤 normalize. 구현 단순하지만 global traffic + sync 비용이 큼.",
    },
    {
      version: "v2.0",
      tag: "Fused Reduce + Normalize (Welford/Online)",
      throughput: "—",
      description:
        "한 패스에서 mean/var(또는 Welford) 누적 후 normalize까지 fusion. DRAM 왕복 감소 + latency 개선.",
    },
    {
      version: "v3.0",
      tag: "Vectorized IO + Warp Reduce Specialization",
      throughput: "—",
      description:
        "float4/half2 로드 + warp-level reduction으로 reduce 비용 축소. (가능하면) affine까지 fuse.",
    },
  ],

  profiling_report: {
    // ncu 수치 생기면 채우면 됨
    SM_Occupancy: "—",
    DRAM_Throughput: "—",
    L1_Cache_Hit_Rate: "—",
    Warp_Execution_Efficiency: "—",
  },

  analysis:
    "FP32는 round-off 수준 오차(<=2.861e-06). FP16은 reduction(Σdy, Σdy·xhat) 누적오차 + affine scaling으로 BWD에서 worst가 커짐(<=1.511e-02). 현재 구현은 2D (M,N)만 지원하며 3D 입력은 NotImplemented로 거부.",

  // 바인딩/정확도 검증 결과 요약
  tests: {
    schema: {
      tag4: "LNEP",
      schema_id_hex: "0x50454e4c",
      payload: "float eps (little-endian)",
      ops: [
        { name: "LayerNormFwd", enum: 13 },
        { name: "LayerNormBwd", enum: 14 },
      ],
    },

    summary:
      "CUDA binding probe vs torch reference. FP32: ~1e-6, FP16: worst 1.511e-2 (BWD affine, M=64 N=256).",

    positive: [
      // ---------- FWD ----------
      { phase: "FWD", dtype: "fp32", affine: false, M: 8, N: 128, eps: 1e-5, max_abs_delta: 2.384e-7 },
      { phase: "FWD", dtype: "fp32", affine: false, M: 64, N: 256, eps: 1e-5, max_abs_delta: 4.768e-7 },
      { phase: "FWD", dtype: "fp32", affine: false, M: 7, N: 33, eps: 1e-5, max_abs_delta: 2.384e-7 },

      { phase: "FWD", dtype: "fp32", affine: true, M: 8, N: 128, eps: 1e-5, max_abs_delta: 7.153e-7 },
      { phase: "FWD", dtype: "fp32", affine: true, M: 64, N: 256, eps: 1e-5, max_abs_delta: 9.537e-7 },
      { phase: "FWD", dtype: "fp32", affine: true, M: 7, N: 33, eps: 1e-5, max_abs_delta: 9.537e-7 },

      { phase: "FWD", dtype: "fp16", affine: false, M: 8, N: 128, eps: 1e-5, max_abs_delta: 1.953e-3 },
      { phase: "FWD", dtype: "fp16", affine: false, M: 64, N: 256, eps: 1e-5, max_abs_delta: 3.906e-3 },
      { phase: "FWD", dtype: "fp16", affine: false, M: 7, N: 33, eps: 1e-5, max_abs_delta: 1.953e-3 },

      { phase: "FWD", dtype: "fp16", affine: true, M: 8, N: 128, eps: 1e-5, max_abs_delta: 3.906e-3 },
      { phase: "FWD", dtype: "fp16", affine: true, M: 64, N: 256, eps: 1e-5, max_abs_delta: 7.812e-3 },
      { phase: "FWD", dtype: "fp16", affine: true, M: 7, N: 33, eps: 1e-5, max_abs_delta: 3.906e-3 },

      // ---------- BWD ----------
      { phase: "BWD", dtype: "fp32", affine: false, M: 8, N: 128, eps: 1e-5, max_abs_delta: 2.384e-7 },
      { phase: "BWD", dtype: "fp32", affine: false, M: 64, N: 256, eps: 1e-5, max_abs_delta: 4.768e-7 },
      { phase: "BWD", dtype: "fp32", affine: false, M: 7, N: 33, eps: 1e-5, max_abs_delta: 2.384e-7 },

      { phase: "BWD", dtype: "fp32", affine: true, M: 8, N: 128, eps: 1e-5, max_abs_delta: 9.537e-7 },
      { phase: "BWD", dtype: "fp32", affine: true, M: 64, N: 256, eps: 1e-5, max_abs_delta: 2.861e-6 },
      { phase: "BWD", dtype: "fp32", affine: true, M: 7, N: 33, eps: 1e-5, max_abs_delta: 9.537e-7 },

      { phase: "BWD", dtype: "fp16", affine: false, M: 8, N: 128, eps: 1e-5, max_abs_delta: 1.953e-3 },
      { phase: "BWD", dtype: "fp16", affine: false, M: 64, N: 256, eps: 1e-5, max_abs_delta: 1.953e-3 },
      { phase: "BWD", dtype: "fp16", affine: false, M: 7, N: 33, eps: 1e-5, max_abs_delta: 1.953e-3 },

      { phase: "BWD", dtype: "fp16", affine: true, M: 8, N: 128, eps: 1e-5, max_abs_delta: 3.486e-3 },
      {
        phase: "BWD",
        dtype: "fp16",
        affine: true,
        M: 64,
        N: 256,
        eps: 1e-5,
        max_abs_delta: 1.511e-2,
        note: "worst (reduction 누적오차 + gamma 스케일 증폭)",
      },
      { phase: "BWD", dtype: "fp16", affine: true, M: 7, N: 33, eps: 1e-5, max_abs_delta: 2.006e-3 },
    ],

    worst: {
      max_abs_delta: 0.015108108520507812,
      case: { phase: "BWD", dtype: "fp16", affine: true, M: 64, N: 256, eps: 1e-5 },
    },

    negative: [
      {
        name: "wrong rank (3D input)",
        op: "LayerNormFwd",
        input_rank: 3,
        expected_status: "NotImplemented",
        got_status: "NotImplemented",
      },
    ],

    notes: [
      "실제 시스템 경로 검증용으로: FWD op가 생성한 mean/rstd를 그대로 BWD op 입력으로 넘기는 e2e 테스트 케이스를 추가하는 걸 권장.",
      "FP16 비교는 절대오차(max abs)뿐 아니라 상대오차(max rel)도 같이 기록하면 케이스 해석이 쉬움.",
    ],
  },
};
