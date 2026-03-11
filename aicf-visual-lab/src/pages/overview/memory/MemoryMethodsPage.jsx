import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Database,
  ArrowUpRight,
  HardDrive,
  Menu,
} from "lucide-react";

import { memoryMethodCatalog } from "../../../data/memory/methodCatalog";
import MemorySidebar from "../../../components/layout/MemorySidebar.jsx";

export default function MemoryMethodsPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <MemorySidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.0.6 Lab-Ready"
      />

      <main className="flex-1 flex flex-col min-w-0 font-sans">
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-5 py-4">
            <Link to="/" className="flex items-center gap-2">
              <div className="bg-emerald-600 p-2 rounded-xl">
                <HardDrive size={18} className="text-white" />
              </div>
              <div className="leading-none">
                <div className="font-black text-emerald-400 tracking-tight">
                  AICF MEMORY
                </div>
                <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                  v1.0.6 Lab-Ready
                </div>
              </div>
            </Link>

            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200 active:scale-95 transition"
              aria-label="Open sidebar"
              type="button"
            >
              <Menu size={18} />
            </button>
          </div>
        </header>

        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-14 pb-32 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          <div className="max-w-6xl mx-auto space-y-20 animate-in fade-in duration-700">
            {/* Hero Section */}
            <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-16 shadow-2xl relative overflow-hidden">
              <div className="absolute -top-10 -right-10 text-[140px] font-black text-emerald-500/5 pointer-events-none">
                MCIR
              </div>

              <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
                <Database size={16} /> Memory Optimization Pattern Catalog
              </div>

              <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                Memory Optimization <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
                  Patterns
                </span>
              </h1>

              <p className="mt-8 text-slate-400 text-lg sm:text-xl leading-relaxed max-w-3xl font-light">
                AICF는 메모리 최적화를 개별 커널 트릭의 모음이 아니라,
                연산의 수학적 성질과 데이터 이동 구조로 분해해 다룹니다.
                <br />
                아래의 네 가지 method는 이러한 관점을 바탕으로 정리한
                핵심 memory optimization pattern catalog입니다.
              </p>
            </section>

            {/* Catalog Intro */}
            <section className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              <div className="rounded-[2rem] border border-slate-800 bg-[#1e293b]/50 p-8">
                <div className="text-xs font-black uppercase tracking-[0.25em] text-emerald-400 mb-4">
                  01. Mathematical Property
                </div>
                <p className="text-sm leading-relaxed text-slate-400">
                  각 method는 단순한 구현 기법이 아니라, mergeable state,
                  rescaling invariance, recompute safety, tile closure와 같은
                  수학적 또는 구조적 성질에서 출발합니다.
                </p>
              </div>

              <div className="rounded-[2rem] border border-slate-800 bg-[#1e293b]/50 p-8">
                <div className="text-xs font-black uppercase tracking-[0.25em] text-emerald-400 mb-4">
                  02. Compiler Transformation
                </div>
                <p className="text-sm leading-relaxed text-slate-400">
                  AICF는 이러한 성질을 compiler-recognizable property로 다루며,
                  multi-pass reduction, materialized intermediate, naive loop
                  ordering을 더 효율적인 실행 구조로 변환합니다.
                </p>
              </div>

              <div className="rounded-[2rem] border border-slate-800 bg-[#1e293b]/50 p-8">
                <div className="text-xs font-black uppercase tracking-[0.25em] text-emerald-400 mb-4">
                  03. Hardware Realization
                </div>
                <p className="text-sm leading-relaxed text-slate-400">
                  최종 목적은 HBM traffic을 줄이고 on-chip reuse를 높이며,
                  실제 하드웨어에서 memory-bound bottleneck을 완화할 수 있는
                  kernel execution structure를 만드는 데 있습니다.
                </p>
              </div>
            </section>

            {/* Method Grid */}
            <section className="space-y-8">
              <div className="flex items-center gap-3 text-emerald-400">
                <Database size={24} />
                <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                  Core Memory Optimization Patterns
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {memoryMethodCatalog.map((method) => {
                  const Icon = method.icon;

                  return (
                    <Link
                      key={method.id}
                      to={`/memory/methods/${method.id}`}
                      className={`group relative p-8 rounded-[2rem] bg-[#1e293b]/40 border ${method.color} transition-all duration-500 hover:-translate-y-2`}
                    >
                      <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Icon className={method.iconColor} size={32} />
                      </div>

                      <div className="relative z-10">
                        <div className="mb-6 p-4 w-fit rounded-2xl bg-slate-900/50 border border-slate-700 group-hover:border-slate-500 transition-colors">
                          <Icon className={method.iconColor} size={32} />
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
                            <span
                              key={tag}
                              className="px-3 py-1 rounded-full bg-slate-900 text-[10px] font-bold text-slate-500 border border-slate-800"
                            >
                              #{tag}
                            </span>
                          ))}
                        </div>

                        <div className="flex items-center gap-2 text-emerald-400 font-black text-xs uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-all translate-x-[-10px] group-hover:translate-x-0">
                          Explore Technical Details <ArrowUpRight size={14} />
                        </div>
                      </div>
                    </Link>
                  );
                })}
              </div>
            </section>

            {/* Unifying Philosophy */}
            <section className="bg-emerald-600/5 border border-emerald-500/20 rounded-[3rem] p-12">
              <div className="max-w-4xl mx-auto text-center">
                <h2 className="text-2xl font-black text-white uppercase mb-6">
                  A Unified View of Memory Optimization
                </h2>

                <p className="text-slate-400 text-sm sm:text-base leading-relaxed">
                  이 네 가지 method는 서로 다른 최적화 기법처럼 보이지만,
                  실제로는 모두 같은 질문에 답합니다.
                  어떤 연산은 streaming 가능하고,
                  어떤 intermediate는 다시 계산할 수 있으며,
                  어떤 working set은 온칩에 머무를 수 있는가.
                  <br className="hidden sm:block" />
                  <br className="hidden sm:block" />
                  AICF는 이러한 판단을 개별 구현 트릭이 아니라
                  compiler-recognizable property로 다루는 것을 목표로 합니다.
                  즉, 메모리 최적화는 사후적인 커널 미세조정이 아니라,
                  연산의 구조를 다시 표현하고 lowering하는 방식의 문제로 봅니다.
                </p>
              </div>
            </section>

            {/* Mapping Section */}
            <section className="space-y-8">
              <div className="flex items-center gap-3 text-emerald-400">
                <Database size={24} />
                <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                  Pattern-to-Kernel Mapping
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div className="rounded-[2rem] border border-slate-800 bg-[#1e293b]/50 p-8">
                  <h3 className="text-lg font-black text-white mb-4">
                    Statistical Streaming
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-400">
                    Online Reducible Norm은 LayerNorm, RMSNorm과 같은 통계 기반
                    normalization kernel에서 나타나는 single-pass reduction
                    패턴을 설명합니다.
                  </p>
                </div>

                <div className="rounded-[2rem] border border-slate-800 bg-[#1e293b]/50 p-8">
                  <h3 className="text-lg font-black text-white mb-4">
                    Weighted Streaming
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-400">
                    Streaming Weighted Reduction은 FlashAttention과 같은
                    attention kernel에서 나타나는 online softmax 기반 streaming
                    reduction 구조를 일반화합니다.
                  </p>
                </div>

                <div className="rounded-[2rem] border border-slate-800 bg-[#1e293b]/50 p-8">
                  <h3 className="text-lg font-black text-white mb-4">
                    Recomputation Tradeoff
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-400">
                    Rematerialization은 activation checkpointing이나 fused
                    epilogue처럼 intermediate storage를 줄이기 위해 일부 계산을
                    다시 수행하는 전략과 연결됩니다.
                  </p>
                </div>

                <div className="rounded-[2rem] border border-slate-800 bg-[#1e293b]/50 p-8">
                  <h3 className="text-lg font-black text-white mb-4">
                    On-Chip Residency
                  </h3>
                  <p className="text-sm leading-relaxed text-slate-400">
                    Tile-Compatible Compute는 GEMM, convolution, attention과
                    같이 working set의 온칩 체류성과 local reuse가 성능의 핵심이
                    되는 커널 구조를 설명합니다.
                  </p>
                </div>
              </div>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}