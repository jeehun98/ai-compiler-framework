// src/data/relu.js

export const reluData = {
  id: "ReLU",
  category: "비선형 게이팅 / 반공간 정류 (Nonlinear Gating)",

  descriptions: {
    essence: "음수 영역의 노이즈를 '0'으로 침묵(Silence)시키고 유의미한 신호만 통과시켜, 네트워크에 비선형성과 희소성(Sparsity)을 부여합니다.",
    strategy: "음수 영역의 정보를 소거하는 게이팅 로직을 선행 연산의 쓰기 단계에 주입(Injection)하여, 별도의 커널 실행 없이 즉각적인 희소성을 확보합니다.",
    hardware: "분기 예측 실패(Branch Divergence)를 방지하기 위해 조건부 이동 명령어(CMOV)나 Max PTX 명령어를 사용하여 단일 사이클에 처리합니다."
  },

  canonical: {
    formula: "y_i = \\max(0, x_i)",
    shapes: {
      x: "M x N (Element-wise)",
      y: "M x N (Rectified)",
    },
    interpretation: {
      x: "판단 전 신호 (Pre-activation Evidence)",
      y: "정제된 신호 (Gated Signal)",
      "0": "결정 경계 (Decision Boundary)",
      "Dead Zone": "정보 소거 영역 (Negative Half-Space)"
    },
  },

  semantics: {
    thesis: "음수 영역의 모호한 신호를 0으로 '침묵(Silence)'시키고, 양수 영역의 신호는 선형적으로 통과시켜 의미론적 희소성(Sparsity)을 부여하는 연산자",

    axes: {
      N: { name: "Neurons", role: "독립적 의사결정 단위" },
      Boundary: { name: "Zero Hyperplane", role: "활성/비활성 분기점" },
    },

    invariants: [
      {
        id: "INV_NONNEGATIVITY",
        name: "비음수 계약 (Non-Negativity Contract)",
        metric: "y >= 0",
        threshold: "Strict",
        allows: ["Unsigned Int 최적화", "메모리 압축"],
      },
      {
        id: "INV_LINEARITY_POS",
        name: "양수 선형성 (Positive Linearity)",
        metric: "if x > 0 then y = x",
        threshold: "Why distortion is bad",
        allows: ["Gradient 소실 방지"],
      },
      {
        id: "INV_SPARSITY_PATTERN",
        name: "희소성 패턴 (Sparsity Pattern)",
        metric: "Zero Ratio",
        threshold: "구조적 가지치기(Pruning) 가능성 진단",
        allows: ["Zero-Skipping 연산", "희소 행렬 변환"],
      },
    ],

    sensitivity: {
      downstream: [
        {
          name: "죽은 뉴런 (Dead Neurons)",
          rule: "특정 채널이 지속적으로 0을 출력하면(Dead), 해당 경로는 연산 자원 낭비임",
          hint: "채널 프루닝 (Channel Pruning) 후보",
        },
        {
          name: "부호 비트 생략",
          rule: "출력이 무조건 양수이므로, 후속 연산에서 부호 비트(Sign Bit)를 무시하거나 압축 가능",
          hint: "데이터 타입 최적화 (Unsigned Int8 Quantization)",
        },
      ],
    },
  },

  lowering: {
    chosen: {
      variant: "Fused_Epilogue_ReLU",
      reason: [
        "메모리 대역폭 절약: 데이터를 읽고 쓰는 비용이 연산 비용보다 큼",
        "커널 융합: 이전 연산(Conv/GEMM/Add)의 데이터를 레지스터에 저장하기 직전에 처리",
        "제로 코스트: 융합 시 물리적인 실행 시간 증가가 거의 없음",
      ],
      applied_rewrites: ["Epilogue Fusion", "Bitmask Generation (for Sparsity)"],
    },
  },

  kernel: {
    strategy: "Epilogue Injection (Zero-Cost)",
    details: [
      { technique: "Epilogue Fusion", semantic_link: "메모리 쓰기(Store) 직전 레지스터 단계에서 ALU 연산 수행" },
      { technique: "Predicate Guard", semantic_link: "분기 예측(Branch Prediction) 없이 조건부 이동(CMOV) 명령어 사용" },
      { technique: "Sparsity Bitmask", semantic_link: "0이 아닌 위치를 1비트로 마킹하여 후속 연산 가속 지원" },
    ],
    metrics: {
      memory_reuse: "Optimal (Fused)",
      throughput: "System Peak",
      occupancy: "N/A (Piggybacked)"
    },
  },

  costModel: {
    semanticLoss: "\\mathcal{L}_{relu} = \\lambda \\cdot \\text{InformationLoss}(x<0)",
    weights_hint: {
      default: { info_loss: 0.0 } // ReLU의 정보 손실은 의도된 것임
    },
    metrics: {
      sparsity_ratio: "~50% (Avg)",
      active_neurons: "High Importance"
    }
  },

  performance: {
    latency: {
      pytorch: 0.05, // 단독 커널 실행 시
      torch_compile: 0.02,
      ours: 0.00 // 융합되어 물리적 실행 시간 0ms
    }
  },

  cudaCode: `// AICF: Fused ReLU (Epilogue)
// Inside GEMM/Conv Kernel:
float acc = ...; // Compute Result
acc = fmaxf(acc, 0.0f); // ReLU (Zero-Cost instruction)
*output = acc; // Store only positive result`
};