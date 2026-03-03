export const gemmEpilogueData = {
  id: "GEMM_EPILOGUE",
  category: "복합 선형 변환 (Fused Linear Transformation)",

  descriptions: {
    essence: "GEMM의 수치적 결과를 의미론적 판단(활성화/편향) 단계와 결합하여, 데이터를 '완성된 특징' 상태로 한 번에 출력하는 고효율 연산자입니다.",
    strategy: "Register Level Fusion을 통해 Bias 가산과 ReLU 필터링을 하드웨어 파이프라인 내부에서 처리함으로써, DRAM 대역폭 낭비를 원천 차단합니다.",
    hardware: "WMMA(Tensor Core)의 Accumulator 조각(Fragment)이 공유 메모리로 나가기 직전, CUDA Core의 산술 연산 유닛을 통해 Epilogue 로직을 주입합니다."
  },

  canonical: {
    // 수식 수정: GEMM 표준 수식 C = alpha(A*B) + beta*C 에 Bias와 ReLU를 결합한 형태
    formula: "Y = \\max(0, \\alpha(A \\times B) + \\beta C + \\text{Bias})",
    shapes: { 
      A: "M x K", 
      B: "K x N", 
      C: "M x N",
      Bias: "N (Vector)", 
      Y: "M x N" 
    },
    interpretation: {
      A: "입력 특징 (질의)",
      B: "선형 투영 가중치 (가설)",
      C: "누적용 베이스 행렬 (Accumulator Base)",
      Bias: "특징 활성화 임계값 조정 (Prior)",
      Y: "정제된 출력 특징 (Final Features)"
    }
  },

  semantics: {
    thesis: "투영된 신호를 편향(Bias)으로 교정하고 비선형성(ReLU)을 부여하여 정보의 유효성을 확정하는 연산",
    
    invariants: [
      {
        id: "INV_EPILOGUE_FUSION",
        name: "원자적 상태 전환 (Atomic State Transition)",
        rule: "GEMM 결과와 Epilogue는 외부에서 관찰 불가능한 하나의 트랜잭션으로 처리되어야 함",
        benefit: "L2 캐시 오염 방지 및 메모리 트래픽 50% 절감"
      },
      {
        id: "INV_RELU_SPARSITY",
        name: "희소성 전파 (Sparsity Propagation)",
        metric: "Zero-value Ratio",
        rule: "ReLU에 의해 0이 된 데이터는 후속 연산의 연산 비용 절감 힌트로 활용",
        allows: ["Sparse-aware Optimization"]
      }
    ],

    sensitivity: {
      precision: {
        f16_tc: "Tensor Core 사용 시 Accumulator는 f32로 유지하여 Bias 합산 시 수치적 정밀도 손실 방지",
        f32_naive: "대형 행렬에서 정밀도가 중요할 경우 사용하며, 병렬 리덕션 효율에 집중"
      }
    }
  },

  lowering: {
    chosen: {
      variant: "WMMA_F16_BiasReLU_Fused",
      reason: [
        "NVIDIA Tensor Core 활용을 통한 연산 밀도 극대화",
        "Bias가 Vector 형태(N)이므로 Warp 내 방송(Broadcast) 로딩 최적화 가능",
        "ReLU의 단순 분기 로직을 인라인 함수화하여 파이프라인 스톨 최소화"
      ],
      applied_rewrites: [
        "Shared Memory Double Buffering",
        "Register-file Epilogue Injection",
        "Warp-level Synchronization Minimization"
      ]
    }
  },

  kernel_analysis: {
    logic: [
      { 
        stage: "Load & Compute", 
        detail: "wmma::load_matrix_sync 및 mma_sync를 통한 하드웨어 가속 연산" 
      },
      { 
        stage: "Epilogue Fusion", 
        detail: "wmma::store_matrix_sync 직전 Accumulator Fragment 단계에서 Bias 가산 및 ReLU 적용" 
      },
      { 
        stage: "Backward (dBias)", 
        detail: "dY와 Y(Mask)를 결합하여 Warp Shuffle 기반의 고속 컬럼 리덕션 수행" 
      }
    ],
    efficiency_metrics: {
      theoretical_vram_saved: "M * N * sizeof(dtype) (중간 결과 C의 쓰기/읽기 생략)",
      compute_intensity: "High (Arithmetic Intensity 증가)"
    }
  }
};