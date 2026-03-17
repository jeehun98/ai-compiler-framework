import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Cpu,
  ArrowRight,
  Sparkles,
  Waypoints,
  Menu,
  Binary,
  Shapes,
  Gauge,
  ShieldCheck,
  Boxes,
} from "lucide-react";
import AppSidebar from "../../components/layout/ComputeSidebar.jsx";

export default function ComputeOverviewPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <AppSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.1.0 Semantic"
      />

      <main className="flex-1 flex flex-col min-w-0">
        {/* Mobile Header */}
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-blue-400 tracking-tighter uppercase">
              AICF Compute
            </div>
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200"
              aria-label="Open sidebar"
            >
              <Menu size={20} />
            </button>
          </div>
        </header>

        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-16">
          {/* Hero Section */}
          <section className="bg-gradient-to-br from-[#1e293b] to-[#0f172a] border border-slate-800 rounded-[2.5rem] p-8 sm:p-12 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[140px] sm:text-[160px] font-black text-blue-500/5 pointer-events-none tracking-tighter uppercase">
              Semantic
            </div>

            <div className="flex items-center gap-2 text-blue-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <Cpu size={16} /> Meaning to Computational Form
            </div>

            <h1 className="mt-6 text-4xl sm:text-6xl font-black tracking-tight leading-[1.08] text-white">
              수학적 본질이
              <br />
              <span className="text-blue-500">계산 가능한 구조</span>로 전개되는 공간
            </h1>

            <p className="mt-6 max-w-3xl text-slate-400 text-base sm:text-lg leading-relaxed">
              Compute 레이어는 연산을 단순한 커널 호출로 보지 않습니다.
              <span className="text-slate-100 font-semibold italic">
                {" "}
                연산의 수학적 정의(Theory)
              </span>
              에서 시작해, 그 의미를 훼손하지 않는
              <span className="text-slate-100 font-semibold italic">
                {" "}
                다양한 계산 구조(Ops)
              </span>
              로 전개합니다. 그리고 마지막에는 실제 런타임 조건 아래에서 가장
              경제적인 실행 경로를 선택합니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/compute/theory"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-blue-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-blue-500 transition-all active:scale-95"
              >
                Theory Specs 탐색 <ArrowRight size={18} />
              </Link>

              <Link
                to="/compute/ops"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                Ops Explorer 보기
              </Link>
            </div>
          </section>

          {/* Core Layers */}
          <section className="space-y-8">
            <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
              <Waypoints size={16} /> Two Layers of Compute
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
              {/* Theory Layer */}
              <div className="bg-[#1e293b]/60 border border-slate-800 rounded-[2rem] p-8 hover:border-blue-500/30 transition group">
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-400 mb-6">
                  <Binary size={24} />
                </div>

                <h3 className="text-2xl font-black text-white">
                  Theory: The Meaning
                </h3>

                <p className="mt-4 text-slate-400 leading-relaxed text-[15px]">
                  연산이 보존해야 하는 <strong>수학적 불변성</strong>과{" "}
                  <strong>동치 조건</strong>을 정의합니다. 하드웨어나 구현 제약을
                  잠시 뒤로 미루고, 연산의 본질이 무엇인지 먼저 고정합니다.
                </p>

                <ul className="mt-6 space-y-2 text-sm text-slate-500 font-mono">
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-blue-500" />
                    Canonical Formulas
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-blue-500" />
                    Geometric Interpretation
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-blue-500" />
                    Invariant Constraints
                  </li>
                </ul>
              </div>

              {/* Ops Layer */}
              <div className="bg-[#1e293b]/60 border border-slate-800 rounded-[2rem] p-8 hover:border-emerald-500/30 transition group">
                <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-400 mb-6">
                  <Shapes size={24} />
                </div>

                <h3 className="text-2xl font-black text-white">
                  Ops: The Form
                </h3>

                <p className="mt-4 text-slate-400 leading-relaxed text-[15px]">
                  Theory에서 고정된 의미를 <strong>실제 계산 가능한 형태</strong>
                  로 전개합니다. 동일한 의미를 유지하면서도 fusion, tiling,
                  tensor core 같은 가속 구조에 맞게 변환 가능한 후보들을
                  구성합니다.
                </p>

                <ul className="mt-6 space-y-2 text-sm text-slate-500 font-mono">
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-emerald-500" />
                    Lowering Candidates
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-emerald-500" />
                    Operator Fusion Space
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-emerald-500" />
                    Semantic Cost Models
                  </li>
                </ul>
              </div>
            </div>
          </section>

          {/* Meaning to Realization */}
          <section className="bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8 sm:p-10 relative overflow-hidden">
            <div className="text-center max-w-2xl mx-auto mb-12">
              <h2 className="text-3xl font-black text-white italic tracking-tight">
                "From Meaning to Realization"
              </h2>
              <p className="mt-4 text-slate-500 leading-relaxed">
                우리가 연산을 다루는 방식은 단순한 구현이 아니라, 의미에서
                형태로, 형태에서 실행으로 이어지는 전개입니다.
              </p>
            </div>

            <div className="flex flex-col md:flex-row items-center justify-between gap-6 max-w-4xl mx-auto">
              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-blue-900/30 border border-blue-500/50 text-blue-400 font-bold">
                  Theory
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Semantic Anchor
                </span>
              </div>

              <ArrowRight className="text-slate-700 hidden md:block" />

              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-emerald-900/30 border border-emerald-500/50 text-emerald-400 font-bold">
                  Ops
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Computational Form
                </span>
              </div>

              <ArrowRight className="text-slate-700 hidden md:block" />

              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-violet-900/30 border border-violet-500/50 text-violet-400 font-bold">
                  Runtime
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Path Binding
                </span>
              </div>

              <ArrowRight className="text-slate-700 hidden md:block" />

              <div className="flex flex-col items-center grayscale opacity-70">
                <div className="px-6 py-3 rounded-full bg-slate-800 border border-slate-700 text-slate-400 font-bold">
                  Realization
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Kernel / Memory
                </span>
              </div>
            </div>
          </section>

          {/* Runtime Bridge Section */}
          <section className="space-y-10 py-2">
            <div className="flex items-center gap-2 text-violet-400 font-black uppercase tracking-widest text-xs">
              <Gauge size={16} /> Runtime Selection Layer
            </div>

            <div className="flex flex-col lg:flex-row gap-10 items-start">
              <div className="lg:w-1/2 space-y-6">
                <h2 className="text-4xl font-black tracking-tight text-white leading-tight">
                  동일한 의미를 유지한 채,
                  <br />
                  <span className="text-violet-400">
                    런타임에서 가장 가벼운 경로
                  </span>
                  를 선택하다
                </h2>

                <p className="text-slate-400 text-lg leading-relaxed">
                  Compute는 Theory와 Ops에서 후보를 만드는 데서 멈추지
                  않습니다. 입력 shape, 메모리 상태, 장치 자원에 따라 현재
                  시점에서 가장 효율적인 실행 경로를 선택합니다. 핵심은 단순히
                  빠른 경로가 아니라,
                  <span className="text-slate-100 font-semibold">
                    {" "}
                    불변성을 깨지 않는 가장 경제적인 경로
                  </span>
                  를 찾는 데 있습니다.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="flex items-center gap-2 text-violet-400 font-black text-lg">
                      <ShieldCheck size={18} />
                      Guard
                    </div>
                    <div className="mt-2 text-slate-500 text-xs uppercase font-bold tracking-wider">
                      Invariant Safety
                    </div>
                  </div>

                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="flex items-center gap-2 text-violet-400 font-black text-lg">
                      <Boxes size={18} />
                      Variant
                    </div>
                    <div className="mt-2 text-slate-500 text-xs uppercase font-bold tracking-wider">
                      Candidate Kernel
                    </div>
                  </div>

                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="flex items-center gap-2 text-violet-400 font-black text-lg">
                      <Cpu size={18} />
                      Binding
                    </div>
                    <div className="mt-2 text-slate-500 text-xs uppercase font-bold tracking-wider">
                      Runtime Parameters
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:w-1/2 w-full bg-[#111827] border border-slate-800 rounded-[2.5rem] p-8 relative overflow-hidden group">
                <div className="absolute inset-0 bg-violet-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />

                <div className="relative space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-4 text-xs font-mono text-slate-500 uppercase tracking-widest">
                    <span>Runtime Path Selection</span>
                    <span className="text-violet-500/60">Guarded Execute</span>
                  </div>

                  <div className="space-y-4">
                    {[
                      {
                        label: "Invariant Check",
                        detail: "의미 보존 조건을 만족하는 경로인지 확인",
                        color: "bg-blue-500",
                      },
                      {
                        label: "Path Search",
                        detail: "현재 입력과 장치 상태에 맞는 후보 경로 선택",
                        color: "bg-violet-500",
                      },
                      {
                        label: "Parameter Binding",
                        detail: "tile, schedule, variant를 실행 시점에 바인딩",
                        color: "bg-emerald-500",
                      },
                    ].map((step) => (
                      <div key={step.label} className="flex items-center gap-4">
                        <div className={`w-2 h-12 ${step.color} rounded-full`} />
                        <div>
                          <div className="text-white font-bold text-sm">
                            {step.label}
                          </div>
                          <div className="text-slate-500 text-xs font-mono mt-0.5">
                            {step.detail}
                          </div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="pt-4">
                    <Link
                      to="/compute/pipeline"
                      className="flex items-center justify-center w-full py-4 rounded-xl border border-violet-500/30 text-violet-400 font-bold text-xs uppercase tracking-widest hover:bg-violet-500 hover:text-[#0b1120] transition-all"
                    >
                      Dynamic Execution 과정 상세 보기
                    </Link>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}