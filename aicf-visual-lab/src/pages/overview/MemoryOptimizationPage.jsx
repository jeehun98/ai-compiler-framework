import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  HardDrive,
  ArrowRight,
  Layers,
  Waypoints,
  ShieldCheck,
  Boxes,
  Sparkles,
  Menu,
  Workflow,
  Database,
  Zap,
} from "lucide-react";
import MemorySidebar from "../../components/layout/MemorySidebar.jsx";

export default function MemoryOptimizationPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden font-sans">
      <MemorySidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.0.6 Lab-Ready"
      />

      <main className="flex-1 flex flex-col min-w-0">
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-emerald-400 tracking-tighter uppercase text-xl">
              AICF MEMORY
            </div>
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200"
              type="button"
              aria-label="Open sidebar"
            >
              <Menu size={20} />
            </button>
          </div>
        </header>

        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-20">
          {/* HERO SECTION */}
          <section className="bg-gradient-to-br from-[#1e293b] to-[#0f172a] border border-slate-800 rounded-[3rem] p-10 md:p-16 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[180px] font-black text-emerald-500/5 pointer-events-none tracking-tighter uppercase select-none">
              Residency
            </div>

            <div className="flex items-center gap-3 text-emerald-400 font-mono text-xs uppercase tracking-[0.4em] font-black mb-8">
              <Zap size={16} className="fill-emerald-400" /> Memory-Centric Execution
            </div>

            <h1 className="text-5xl md:text-7xl font-black tracking-tight leading-[1.05] text-white">
              성능 병목은 종종,
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
                계산보다 데이터 이동에서 시작됩니다
              </span>
            </h1>

            <p className="mt-8 max-w-3xl text-slate-400 text-xl leading-relaxed font-light">
              AICF는 성능 문제를 단순한 FLOPs 부족으로 보지 않고,
              연산 사이에서 반복적으로 발생하는 global memory(HBM) traffic의
              구조로 봅니다.
              <br />
              핵심은 값을 얼마나 오래 on-chip에 유지할 수 있는지,
              어떤 intermediate를 저장하지 않아도 되는지,
              그리고 어떤 reduction을 streaming 형태로 바꿀 수 있는지를
              실행 구조 수준에서 결정하는 것입니다.
            </p>

            <div className="mt-12 flex flex-wrap gap-5">
              <Link
                to="/memory/methods"
                className="inline-flex items-center gap-3 px-8 py-5 rounded-2xl bg-emerald-600 text-white font-bold text-sm uppercase tracking-widest shadow-xl hover:bg-emerald-500 transition-all hover:-translate-y-1 active:scale-95"
              >
                Pattern Catalog 보기 <ArrowRight size={18} />
              </Link>
              <Link
                to="/memory/pipeline"
                className="inline-flex items-center gap-3 px-8 py-5 rounded-2xl border border-slate-700 text-slate-300 font-bold text-sm uppercase tracking-widest hover:bg-slate-800 transition-all"
              >
                Residency Pipeline 보기
              </Link>
            </div>
          </section>

          {/* DOMAIN NARRATIVE */}
          <section id="narrative" className="space-y-10">
            <div className="flex items-center gap-2 text-emerald-500/80 font-black uppercase tracking-[0.3em] text-xs">
              <Waypoints size={16} /> Memory Domain Perspective
            </div>

            <h2 className="text-4xl md:text-5xl font-black tracking-tight text-white">
              Operator 경계를 넘어서,
              <br />
              <span className="text-slate-500 italic">
                physical dataflow를 다시 설계하다
              </span>
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {[
                {
                  k: "Residency",
                  t: "On-Chip Residency 확보",
                  sub: "On-Chip Lifetime",
                  icon: <Sparkles size={20} />,
                  p1: "중간 결과가 너무 일찍 HBM으로 나가면 이후 단계는 다시 같은 데이터를 불러와야 합니다.",
                  p2: "AICF는 register, shared memory, cache 안에서 값이 머무를 수 있는 residency window를 더 길게 만드는 방향으로 실행 구조를 설계합니다.",
                },
                {
                  k: "Boundary",
                  t: "불필요한 경계 제거",
                  sub: "Boundary Elimination",
                  icon: <Layers size={20} />,
                  p1: "논리적으로 분리된 operator 경계는 종종 불필요한 write-read 사이클을 강제합니다.",
                  p2: "AICF는 개별 operator 호출보다, 데이터가 실제로 어떻게 흐르는지를 기준으로 fused execution boundary를 다시 정의합니다.",
                },
                {
                  k: "Planning",
                  t: "자원 제약 기반 계획",
                  sub: "Traffic-Aware Planning",
                  icon: <ShieldCheck size={20} />,
                  p1: "메모리 최적화는 단순한 식 변형이 아니라 working set, bandwidth, on-chip capacity를 동시에 고려하는 planning 문제입니다.",
                  p2: "최적화 가능성은 수식 자체보다도 어떤 데이터 이동을 제거할 수 있는지, 그리고 어떤 state를 local하게 유지할 수 있는지에 의해 결정됩니다.",
                },
              ].map((s) => (
                <div
                  key={s.k}
                  className="bg-[#1e293b]/40 border border-slate-800 rounded-[2.5rem] p-10 shadow-xl hover:border-emerald-500/40 transition-all duration-300 group hover:bg-[#1e293b]/60"
                >
                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[11px] uppercase tracking-[0.25em] font-black group-hover:text-emerald-400 transition-colors">
                    {s.icon} {s.k}
                  </div>
                  <div className="mt-6">
                    <div className="text-white font-black text-2xl tracking-tight leading-tight">
                      {s.t}
                    </div>
                    <div className="text-emerald-500/60 font-mono text-xs font-bold uppercase tracking-wider mt-2">
                      {s.sub}
                    </div>
                  </div>
                  <p className="mt-8 text-slate-400 leading-relaxed text-base font-light">
                    {s.p1}
                  </p>
                  <p className="mt-4 text-slate-500 leading-relaxed text-sm italic border-l border-emerald-500/20 pl-4">
                    {s.p2}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* EXECUTION STRATEGY */}
          <section className="space-y-12 py-10">
            <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
              <Boxes size={16} /> Memory Planning Architecture
            </div>

            <div className="flex flex-col lg:flex-row gap-16 items-center">
              <div className="lg:w-1/2 space-y-8">
                <h2 className="text-4xl md:text-5xl font-black tracking-tight text-white leading-[1.1]">
                  커널을 나누는 대신,
                  <br />
                  <span className="text-emerald-400">데이터 경로를 계획한다</span>
                </h2>

                <p className="text-slate-400 text-xl leading-relaxed font-light">
                  AICF의 메모리 최적화는 단순한 fusion checklist가 아닙니다.
                  핵심은 어떤 값을 언제 생성하고, 얼마나 오래 유지하며,
                  어느 시점에만 global memory에 기록할지를 결정하는
                  execution planner에 있습니다.
                </p>

                <div className="grid grid-cols-2 gap-6 pt-4">
                  <div className="p-6 rounded-[2rem] bg-slate-800/30 border border-slate-700 group hover:border-emerald-500/30 transition-colors">
                    <div className="text-emerald-400 font-black text-3xl mb-1">
                      Reduced
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-[0.2em]">
                      Global Memory Traffic
                    </div>
                  </div>
                  <div className="p-6 rounded-[2rem] bg-slate-800/30 border border-slate-700 group hover:border-blue-500/30 transition-colors">
                    <div className="text-blue-400 font-black text-3xl mb-1">
                      Structured
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-[0.2em]">
                      Fused Execution Path
                    </div>
                  </div>
                </div>
              </div>

              {/* Visual Workflow Card */}
              <div className="lg:w-1/2 w-full bg-[#0b1120] border border-slate-800 rounded-[3rem] p-10 relative overflow-hidden group shadow-3xl">
                <div className="absolute inset-0 bg-emerald-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="space-y-8 relative">
                  <div className="flex items-center justify-between border-b border-slate-800/50 pb-6 text-xs font-mono text-slate-500 uppercase tracking-widest">
                    <span>Memory Execution Workflow</span>
                    <span className="text-emerald-500 font-bold animate-pulse">
                      Planner Active
                    </span>
                  </div>

                  <div className="space-y-6">
                    {[
                      {
                        label: "Working Set and Residency Analysis",
                        detail: "on-chip capacity 안에서 유지 가능한 tile 구조와 working set을 계산합니다.",
                        color: "bg-blue-500",
                      },
                      {
                        label: "Streaming and Fusion Rewrite",
                        detail: "streaming reduction, rematerialization, fusion 가능성을 property 기반으로 판별합니다.",
                        color: "bg-indigo-500",
                      },
                      {
                        label: "Traffic-Aware Kernel Lowering",
                        detail: "필요한 시점에만 global memory write를 허용하도록 execution path를 생성합니다.",
                        color: "bg-emerald-500",
                      },
                    ].map((step, i) => (
                      <div key={i} className="flex items-center gap-6 group/item">
                        <div className={`w-3 h-14 ${step.color} rounded-full`} />
                        <div>
                          <div className="text-white font-black text-lg group-hover/item:text-emerald-300 transition-colors">
                            {step.label}
                          </div>
                          <div className="text-slate-500 text-sm font-mono mt-1 font-medium">
                            {step.detail}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <Link
                    to="/memory/pipeline"
                    className="mt-10 flex items-center justify-center w-full py-5 rounded-2xl bg-slate-900 border border-emerald-500/20 text-emerald-400 font-black text-sm uppercase tracking-[0.2em] hover:bg-emerald-500 hover:text-[#0b1120] transition-all shadow-inner"
                  >
                    Residency Pipeline 상세 보기
                  </Link>
                </div>
              </div>
            </div>
          </section>

          {/* PATTERN CONNECTION */}
          <section className="space-y-10">
            <div className="flex items-center gap-2 text-emerald-500/80 font-black uppercase tracking-[0.3em] text-xs">
              <Database size={16} /> Pattern View
            </div>

            <h2 className="text-4xl md:text-5xl font-black tracking-tight text-white">
              메모리 최적화는 결국,
              <br />
              <span className="text-slate-500 italic">
                구조적 패턴의 식별 문제다
              </span>
            </h2>

            <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-4 gap-6">
              {[
                {
                  title: "Streaming Statistics",
                  body: "일부 statistic reduction은 full materialization 없이 mergeable state로 누적할 수 있습니다.",
                },
                {
                  title: "Weighted Streaming",
                  body: "정규화된 weighted reduction도 rescaling-invariant structure를 가지면 streaming 형태로 바꿀 수 있습니다.",
                },
                {
                  title: "Recomputation Tradeoff",
                  body: "저장보다 재계산이 싼 intermediate는 materialize하지 않고 다시 생성할 수 있습니다.",
                },
                {
                  title: "Tile Residency",
                  body: "working set과 dependency가 tile boundary 안에서 닫히면 on-chip reuse를 극대화할 수 있습니다.",
                },
              ].map((item) => (
                <div
                  key={item.title}
                  className="rounded-[2rem] border border-slate-800 bg-[#1e293b]/40 p-8 hover:border-emerald-500/30 transition-colors"
                >
                  <h3 className="text-lg font-black text-white mb-4">
                    {item.title}
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-400">
                    {item.body}
                  </p>
                </div>
              ))}
            </div>

            <div className="pt-2">
              <Link
                to="/memory/methods"
                className="inline-flex items-center gap-3 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-sm uppercase tracking-widest hover:bg-slate-800 transition-all"
              >
                Pattern Catalog 전체 보기 <ArrowRight size={16} />
              </Link>
            </div>
          </section>

          {/* FINAL STATEMENT */}
          <section className="bg-emerald-950/20 border border-emerald-500/20 rounded-[3rem] p-12 md:p-20 relative overflow-hidden text-center">
            <div className="absolute inset-0 opacity-20 pointer-events-none" />

            <div className="flex justify-center mb-8">
              <div className="px-5 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono text-[10px] uppercase tracking-[0.4em] font-black">
                AICF Memory Position
              </div>
            </div>

            <h3 className="text-4xl md:text-6xl font-black tracking-tighter text-white leading-[1.1]">
              메모리 최적화의 본질은,
              <br />
              <span className="italic text-emerald-400">
                계산을 더 많이 하는 것이 아니라
              </span>
              <br />
              불필요한 이동을 더 적게 만드는 데 있습니다.
            </h3>

            <p className="mt-10 max-w-3xl mx-auto text-slate-400 text-lg leading-relaxed font-light">
              AICF는 operator graph를 그대로 실행하는 대신,
              어떤 값은 local state로 유지하고,
              어떤 intermediate는 제거하거나 다시 계산하며,
              어떤 reduction은 streaming 구조로 바꾸는지 결정합니다.
              <br />
              이 관점에서 메모리 최적화는 사후적인 미세조정보다
              execution structure를 다시 표현하는 문제에 가깝습니다.
            </p>

            <div className="mt-12 flex flex-wrap justify-center gap-4">
              {[
                "On-Chip Residency",
                "Boundary Elimination",
                "Streaming Reduction",
                "Traffic-Aware Planning",
              ].map((tag) => (
                <span
                  key={tag}
                  className="px-5 py-2.5 rounded-xl bg-slate-900/80 border border-slate-800 text-[11px] font-black uppercase tracking-widest text-emerald-400/70"
                >
                  {tag}
                </span>
              ))}
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}