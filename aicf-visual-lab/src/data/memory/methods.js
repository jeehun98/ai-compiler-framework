import {
  Activity,
  Zap,
  RotateCcw,
  Maximize2,
} from "lucide-react";

export const memoryMethods = [
  {
    id: "online-norm",
    label: "Online Reducible Norm",
    category: "Single-Pass Reduction",
    icon: Activity,
    navIcon: Activity,
    iconColor: "text-emerald-400",
    desc:
      "Welford 알고리즘을 활용하여 데이터를 한 번만 읽으면서(Single-pass) 평균과 분산을 계산합니다. HBM 트래픽을 50% 이상 절감합니다.",
    tags: ["Single-pass", "Welford", "Stat-Reduction"],
    color: "border-emerald-500/20 hover:border-emerald-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "weighted-reduction",
    label: "Streaming Weighted Reduction",
    category: "Flash-Style Optimization",
    icon: Zap,
    navIcon: Zap,
    iconColor: "text-amber-400",
    desc:
      "FlashAttention의 핵심 원리를 일반화하여, 가중치가 포함된 합산(Weighted Sum)을 지수적 재조정(Rescaling)을 통해 스트리밍합니다.",
    tags: ["Flash-Attention", "Rescaling", "Softmax-Fusion"],
    color: "border-amber-500/20 hover:border-amber-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "rematerialization",
    label: "Re-materializable Intermediate",
    category: "VRAM Saving Strategy",
    icon: RotateCcw,
    navIcon: RotateCcw,
    iconColor: "text-blue-400",
    desc:
      "메모리 대역폭이 연산 속도보다 느린 병목 구간에서, 중간 값을 저장하지 않고 온칩에서 즉석 재계산하여 메모리 점유율을 극단적으로 낮춥니다.",
    tags: ["Re-compute", "Bandwidth-Aware", "Checkpointing"],
    color: "border-blue-500/20 hover:border-blue-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "tile-compatible",
    label: "Tile-Compatible Compute",
    category: "SRAM Residency Planning",
    icon: Maximize2,
    navIcon: Maximize2,
    iconColor: "text-indigo-400",
    desc:
      "가속기의 SRAM 용량을 고려한 타일링 전략입니다. 데이터가 캐시 라인을 벗어나지 않도록 연산 순서를 재조직하여 체류성(Residency)을 극대화합니다.",
    tags: ["Tiling", "SRAM-Optimization", "L1-Cache"],
    color: "border-indigo-500/20 hover:border-indigo-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
];

export const memoryMethodsMap = Object.fromEntries(
  memoryMethods.map((method) => [method.id, method])
);

export const memoryPhaseLabels = {
  theory: "Math & Logic",
  hardware: "Physical Analysis",
  compiler: "MCIR Implementation",
};