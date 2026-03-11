import React from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  Zap,
  RotateCcw,
  Maximize2,
  ArrowRight,
  Database,
  Cpu,
  Layers,
  ArrowUpRight
} from "lucide-react";

export default function MemoryMethodsPage() {
  // 사이드바와 동일한 데이터 구조
  const methods = [
    {
      id: "online-norm",
      label: "Online Reducible Norm",
      category: "Single-Pass Reduction",
      icon: <Activity className="text-emerald-400" size={32} />,
      desc: "Welford 알고리즘을 활용하여 데이터를 한 번만 읽으면서(Single-pass) 평균과 분산을 계산합니다. HBM 트래픽을 50% 이상 절감합니다.",
      tags: ["Single-pass", "Welford", "Stat-Reduction"],
      color: "border-emerald-500/20 hover:border-emerald-500/50"
    },
    {
      id: "weighted-reduction",
      label: "Streaming Weighted Reduction",
      category: "Flash-Style Optimization",
      icon: <Zap className="text-amber-400" size={32} />,
      desc: "FlashAttention의 핵심 원리를 일반화하여, 가중치가 포함된 합산(Weighted Sum)을 지수적 재조정(Rescaling)을 통해 스트리밍합니다.",
      tags: ["Flash-Attention", "Rescaling", "Softmax-Fusion"],
      color: "border-amber-500/20 hover:border-amber-500/50"
    },
    {
      id: "rematerialization",
      label: "Re-materializable Intermediate",
      category: "VRAM Saving Strategy",
      icon: <RotateCcw className="text-blue-400" size={32} />,
      desc: "메모리 대역폭이 연산 속도보다 느린 병목 구간에서, 중간 값을 저장하지 않고 온칩에서 즉석 재계산하여 메모리 점유율을 극단적으로 낮춥니다.",
      tags: ["Re-compute", "Bandwidth-Aware", "Checkpointing"],
      color: "border-blue-500/20 hover:border-blue-500/50"
    },
    {
      id: "tile-compatible",
      label: "Tile-Compatible Compute",
      category: "SRAM Residency Planning",
      icon: <Maximize2 className="text-indigo-400" size={32} />,
      desc: "가속기의 SRAM 용량을 고려한 타일링 전략입니다. 데이터가 캐시 라인을 벗어나지 않도록 연산 순서를 재조직하여 체류성(Residency)을 극대화합니다.",
      tags: ["Tiling", "SRAM-Optimization", "L1-Cache"],
      color: "border-indigo-500/20 hover:border-indigo-500/50"
    },
  ];

  return (
    <div className="flex-1 overflow-y-auto bg-[#0f172a] text-slate-200 p-6 md:p-12">
      {/* Hero Section */}
      <section className="max-w-6xl mx-auto mb-16">
        <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs uppercase tracking-[0.3em] font-black mb-4">
          <Database size={16} /> Architecture Pillars
        </div>
        <h1 className="text-4xl md:text-6xl font-black text-white tracking-tight mb-6">
          Optimization <br />
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
            Methodologies
          </span>
        </h1>
        <p className="text-slate-400 text-lg max-w-3xl leading-relaxed">
          AICF는 단순한 코드 최적화를 넘어, 하드웨어의 물리적 제약을 수학적 성질로 극복합니다. 
          아래의 4가지 핵심 기법은 메모리 벽(Memory Wall)을 허물고 연산 효율을 물리적 한계치까지 끌어올리는 AICF의 기술적 기둥입니다.
        </p>
      </section>

      {/* Grid Layout */}
      <div className="max-w-6xl mx-auto grid grid-cols-1 md:grid-cols-2 gap-6">
        {methods.map((method) => (
          <Link
            key={method.id}
            to={`/memory/methods/${method.id}`}
            className={`group relative p-8 rounded-[2rem] bg-[#1e293b]/40 border ${method.color} transition-all duration-500 hover:-translate-y-2`}
          >
            {/* Background Decoration */}
            <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
              {method.icon}
            </div>

            <div className="relative z-10">
              <div className="mb-6 p-4 w-fit rounded-2xl bg-slate-900/50 border border-slate-700 group-hover:border-slate-500 transition-colors">
                {method.icon}
              </div>

              <div className="space-y-2 mb-6">
                <span className="text-xs font-black text-emerald-500/70 uppercase tracking-widest">
                  {method.category}
                </span>
                <h3 className="text-2xl font-black text-white group-hover:text-emerald-400 transition-colors">
                  {method.label}
                </h3>
              </div>

              <p className="text-slate-400 text-sm leading-relaxed mb-8 group-hover:text-slate-300 transition-colors">
                {method.desc}
              </p>

              <div className="flex flex-wrap gap-2 mb-8">
                {method.tags.map((tag) => (
                  <span key={tag} className="px-3 py-1 rounded-full bg-slate-900 text-[10px] font-bold text-slate-500 border border-slate-800">
                    #{tag}
                  </span>
                ))}
              </div>

              <div className="flex items-center gap-2 text-emerald-400 font-black text-xs uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-all translate-x-[-10px] group-hover:translate-x-0">
                Explore Tech Spec <ArrowUpRight size={14} />
              </div>
            </div>
          </Link>
        ))}
      </div>

      {/* Philosophy Callout */}
      <section className="max-w-6xl mx-auto mt-20 p-10 rounded-[3rem] border border-slate-800 bg-gradient-to-b from-[#111827] to-[#0f172a] text-center">
        <h2 className="text-2xl font-black text-white mb-4">수학적 원리가 곧 물리적 속도가 됩니다.</h2>
        <p className="text-slate-500 text-sm max-w-2xl mx-auto leading-relaxed italic">
          "모든 최적화는 '값의 성질'을 정의하는 것에서 시작합니다. <br />
          AICF는 연산의 결합 법칙과 단조성을 이용하여 데이터가 칩 밖을 나가지 않아도 되는 수학적 증명을 코드에 주입합니다."
        </p>
      </section>
    </div>
  );
}