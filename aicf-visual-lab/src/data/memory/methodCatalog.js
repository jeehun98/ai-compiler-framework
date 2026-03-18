import { Activity, Zap, RotateCcw, Maximize2 } from "lucide-react";

export const memoryMethodCatalog = [
  {
    id: "online-norm",
    label: "Online Reducible Norm",
    category: "Reduction Structure",
    icon: Activity,
    navIcon: Activity,
    iconColor: "text-emerald-400",
    desc:
      "전체 값을 materialize하지 않고 통계량을 streaming state로 유지하는 reduction 구조입니다. Welford와 같은 알고리즘을 통해 mean/variance를 single-pass로 계산할 수 있으며, multi-pass reduction을 online state update로 변환하여 global memory(HBM) 접근을 최소화합니다.",
    tags: [
      "Online-State",
      "Single-Pass",
      "Mergeable-Statistics",
      "Reduction-Folding",
    ],
    color: "border-emerald-500/20 hover:border-emerald-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "weighted-reduction",
    label: "Streaming Weighted Reduction",
    category: "Reduction Structure",
    icon: Zap,
    navIcon: Zap,
    iconColor: "text-amber-400",
    desc:
      "동적으로 결정되는 weight를 포함한 reduction을 streaming 형태로 재구성하는 구조입니다. FlashAttention의 online softmax 원리를 일반화한 것으로, rescaling을 통해 수치 안정성을 유지하면서 large weighted reduction을 tile-local accumulation으로 변환합니다.",
    tags: [
      "Streaming-Reduction",
      "Rescaling-Invariance",
      "Softmax-Generalization",
      "Dynamic-Weights",
    ],
    color: "border-amber-500/20 hover:border-amber-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "rematerialization",
    label: "Re-materializable Intermediate",
    category: "Value Lifetime",
    icon: RotateCcw,
    navIcon: RotateCcw,
    iconColor: "text-blue-400",
    desc:
      "intermediate 값을 저장하지 않고 필요 시 다시 계산할 수 있는 구조적 성질입니다. value lifetime을 줄여 memory footprint를 감소시키며, recompute를 통해 global memory traffic을 줄이는 방향으로 execution을 재구성할 수 있습니다.",
    tags: [
      "Recompute-Safe",
      "Lifetime-Reduction",
      "Checkpointing",
      "Bandwidth-Tradeoff",
    ],
    color: "border-blue-500/20 hover:border-blue-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
  {
    id: "tile-compatible",
    label: "Tile-Compatible Compute",
    category: "Execution Locality",
    icon: Maximize2,
    navIcon: Maximize2,
    iconColor: "text-indigo-400",
    desc:
      "연산 순서를 재구성하여 working set이 tile 단위로 on-chip memory(shared memory / registers)에 머무를 수 있도록 만드는 구조입니다. dependency와 access pattern을 재배열함으로써 global memory 접근을 줄이고 local reuse를 극대화합니다.",
    tags: [
      "Tile-Closure",
      "On-Chip-Residency",
      "Data-Locality",
      "Schedule-Reordering",
    ],
    color: "border-indigo-500/20 hover:border-indigo-500/50",
    phases: ["theory", "hardware", "compiler"],
  },
];

export const memoryMethodCatalogMap = Object.fromEntries(
  memoryMethodCatalog.map((method) => [method.id, method])
);