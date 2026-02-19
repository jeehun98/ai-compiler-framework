// src/data/batch_norm.js

export const batchNormData = {
  id: "BatchNorm",
  category: "집단 통계 정렬 / 분포 계약 (Collective Distribution Contract)",

  descriptions: {
    essence: "개별 데이터가 아닌 집단(Batch)의 통계를 기준으로 데이터를 재배치하여, 전체적인 학습 분포를 강제로 정렬하는 사회적 계약 연산자입니다.",
    strategy: "학습 시에는 집단 통계 동기화를 가속하고, 추론 시에는 선행 가중치에 수식을 수학적으로 통합(Folding)하여 연산 노드 자체를 소멸시킵니다.",
    hardware: "학습 시에는 Persistent Thread Block을 유지하여 데이터를 재사용하고, 추론 시에는 아예 실행되는 커널이 없는(Zero-Kernel) 상태를 만듭니다."
  },

  canonical: {
    formula: [
      "\\mu_B = \\frac{1}{B} \\sum x_i, \\quad \\sigma^2_B = \\frac{1}{B} \\sum (x_i - \\mu_B)^2",
      "\\hat{x}_i = \\frac{x_i - \\mu_B}{\\sqrt{\\sigma^2_B + \\epsilon}}",
      "y_i = \\gamma \\hat{x}_i + \\beta",
      "\\text{(Inference)} \\quad y = w_{fold} x + b_{fold}"
    ].join("\\\\"),
    shapes: {
      x: "B x C x H x W",
      "\\mu, \\sigma": "1 x C (Channel Stats)",
      "\\gamma, \\beta": "1 x C (Learnable Params)",
      y: "B x C x H x W"
    },
    interpretation: {
      "\\mu, \\sigma": "집단 지성 기준점 (Collective Reference Frame)",
      "\\gamma, \\beta": "고유 표현력 복원 (Affine Restoration)",
      "Running Stats": "추론을 위한 기억 (Inference Bridge)",
      "Folded": "연산 소멸 (Operator Erasure)"
    },
  },

  semantics: {
    thesis: "개별 데이터의 절대적 수치를 무시하고, 집단 내에서의 상대적 위치(Z-score)로 변환하여 학습 궤적을 강제로 안정화하는 연산자",

    axes: {
      C: { name: "Channels", role: "독립적인 정규화 계약 단위" },
      B: { name: "Batch", role: "통계적 유의성을 확보하기 위한 표본 집단" },
    },

    invariants: [
      {
        id: "INV_DIST_STABILITY",
        name: "분포 안정성 (Distribution Stability)",
        metric: "E[y] ≈ 0, Var(y) ≈ 1",
        threshold: "초기 학습 단계 필수 조건",
        allows: ["Learning Rate 증가", "가중치 초기화 민감도 감소"],
      },
      {
        id: "INV_INFERENCE_CONSISTENCY",
        name: "학습-추론 일치성 (Train-Infer Consistency)",
        metric: "Train Stats vs Running Stats",
        threshold: "이동 평균(EMA)의 수렴 보장",
        allows: ["추론 시 Conv-BN 융합(Folding)"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "배치 크기 위기 (Small Batch Crisis)",
          rule: "\\text{배치 크기가 작으면}(B < 8)\\text{ 통계가 부정확해져 학습이 파탄남}",
          hint: "GroupNorm/LayerNorm으로 대체 권장",
        },
        {
          name: "분산 통신 비용 (SyncBN)",
          rule: "멀티 GPU 학습 시, 정확한 통계를 위해 GPU 간 통신(AllReduce)이 필요함",
          hint: "통신-연산 오버랩 (Overlap) 필수",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Training: Fused_SyncBatchNorm | Inference: Folded_Erasure",
      reason: [
        "학습 모드: 메모리 대역폭이 병목이므로 2-Pass 알고리즘을 단일 커널로 융합",
        "멀티 GPU: SyncBatchNorm을 적용하여 전역적(Global) 통계 정확도 확보",
        "추론 모드: 앞단의 Conv 레이어와 수학적으로 합칠 수 있으므로(Folding), 실제 실행 시 연산 제거",
      ],
      applied_rewrites: ["Conv-BN Fusion (Inference)", "Persistent CTA Reduction (Training)"],
    },
  },

  kernel: {
    strategy: "Persistent CTA Reduction & Stat Sync",
    details: [
      { technique: "Persistent Thread Block", semantic_link: "데이터 재로딩 없이 통계 산출 및 정규화 수행" },
      { technique: "Warp-Level AllReduce", semantic_link: "GPU 내부 및 GPU 간 통계 동기화 가속" },
      { technique: "Inference Folding", semantic_link: "수학적 결합을 통해 런타임 연산 비용 0으로 만듦" },
    ],
    metrics: {
      memory_reuse: "High (Persistent)",
      throughput: "Network Bound (SyncBN) / Max (Local)",
      occupancy: "85%"
    },
  },

  costModel: {
    semanticLoss: "\\mathcal{L}_{BN} = \\lambda_{sync} \\cdot \\text{CommCost} + \\mathcal{L}_{drift}",
    weights_hint: {
      default: { sync_overhead: 30.0, stat_error: 50.0 }
    },
    metrics: {
      sync_overhead: "12us (NVLink)",
      folding_gain: "100% (Removed)"
    }
  },

  performance: {
    latency: {
      pytorch: 0.50, // Training 기준
      torch_compile: 0.35,
      ours: 0.15 // Training (Fused)
      // Note: Inference 시 ours는 0.00ms (Folded)
    }
  },

  cudaCode: `// AICF: Fused SyncBatchNorm (Training)
__global__ void batch_norm_train(...) {
  // 1. Local Sum/Square-Sum Reduction (Register)
  // 2. Cross-Device AllReduce (NCCL/NVLink Sync)
  // 3. Compute Global Mean/Var
  // 4. Normalize & Affine Transform (Write Back)
  // 5. Update Running Stats (Side Effect)
}

// Inference: No Kernel (Merged into Conv weights)`
};