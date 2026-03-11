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
      "Welford 알고리즘을 이용해 평균과 분산을 single-pass streaming 방식으로 계산합니다. 기존 multi-pass reduction을 streaming reduction으로 바꾸어 global memory(HBM) 접근을 줄일 수 있습니다.",
    tags: ["Single-pass", "Welford", "Stat-Reduction"],
    color: "border-emerald-500/20 hover:border-emerald-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "weighted-reduction",
    label: "Streaming Weighted Reduction",
    category: "Streaming Reduction",
    icon: Zap,
    navIcon: Zap,
    iconColor: "text-amber-400",
    desc:
      "FlashAttention의 online softmax 원리를 일반화하여 weighted sum을 streaming reduction 형태로 재구성합니다. Rescaling을 통해 수치 안정성을 유지하면서 large reduction을 tile-local accumulation으로 변환합니다.",
    tags: ["Flash-Attention", "Rescaling", "Softmax-Fusion"],
    color: "border-amber-500/20 hover:border-amber-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "rematerialization",
    label: "Re-materializable Intermediate",
    category: "Memory–Compute Tradeoff",
    icon: RotateCcw,
    navIcon: RotateCcw,
    iconColor: "text-blue-400",
    desc:
      "중간 텐서를 저장하지 않고 필요할 때 다시 계산(rematerialization)합니다. 메모리 사용량을 줄이는 대신 일부 연산을 재수행하는 memory–compute tradeoff 전략입니다.",
    tags: ["Re-compute", "Bandwidth-Aware", "Checkpointing"],
    color: "border-blue-500/20 hover:border-blue-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "tile-compatible",
    label: "Tile-Compatible Compute",
    category: "On-Chip Residency Planning",
    icon: Maximize2,
    navIcon: Maximize2,
    iconColor: "text-indigo-400",
    desc:
      "연산 순서를 재구성하여 working set이 on-chip memory(shared memory / registers)에 머무를 수 있도록 설계합니다. 이를 통해 global memory 접근을 줄이고 tile-local computation을 극대화합니다.",
    tags: ["Tiling", "SRAM-Optimization", "L1-Cache"],
    color: "border-indigo-500/20 hover:border-indigo-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
];

export const memoryMethodCatalogMap = Object.fromEntries(
  memoryMethodCatalog.map((method) => [method.id, method])
);