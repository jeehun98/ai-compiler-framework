import { Activity, Zap, RotateCcw, Maximize2 } from "lucide-react";

export const memoryMethodCatalog = [
  {
    id: "online-norm",
    label: "Online Reducible Norm",
    category: "Single-Pass Reduction",
    icon: Activity,
    navIcon: Activity,
    iconColor: "text-emerald-400",
    desc:
      "Welford 알고리즘을 활용하여 데이터를 한 번만 읽으면서 평균과 분산을 계산합니다. HBM 트래픽을 크게 줄일 수 있습니다.",
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
      "FlashAttention의 핵심 원리를 일반화하여 가중 합산을 스트리밍 구조로 재구성합니다.",
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
      "중간 값을 저장하지 않고 필요할 때 다시 계산하여 메모리 점유율과 bandwidth pressure를 낮춥니다.",
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
      "온칩 메모리 체류성을 높이기 위해 연산 순서와 working set을 타일 친화적으로 재구성합니다.",
    tags: ["Tiling", "SRAM-Optimization", "L1-Cache"],
    color: "border-indigo-500/20 hover:border-indigo-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
];

export const memoryMethodCatalogMap = Object.fromEntries(
  memoryMethodCatalog.map((method) => [method.id, method])
);