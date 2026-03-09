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
} from "lucide-react";
import AppSidebar from "../components/AppSidebar.jsx";

export default function MemoryOptimizationPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <AppSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeOpId={null}
        quickOps={["FlashAttention", "LayerNorm", "Softmax", "GEMM"]}
        version="v1.0.4 Stable"
      />

      <main className="flex-1 flex flex-col min-w-0">
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-emerald-400 tracking-tighter uppercase">
              AICF Lab
            </div>
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200"
            >
              <Menu size={20} />
            </button>
          </div>
        </header>

        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-16">
          {/* HERO SECTION */}
          <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-12 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[160px] font-black text-emerald-500/5 pointer-events-none tracking-tighter uppercase">
              Residency
            </div>

            <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <HardDrive size={16} /> Dataflow, Residency & Traffic Control
            </div>

            <h1 className="mt-6 text-5xl sm:text-6xl font-black tracking-tight leading-[1.1] text-white">
              연산보다 더 비싼 것은
              <br />
              데이터를 다시 읽는 일이다
            </h1>

            <p className="mt-6 max-w-3xl text-slate-400 text-lg leading-relaxed">
              AICF는 성능 병목을 FLOPs가 아니라
              <span className="text-slate-100 font-semibold italic">
                {" "}
                데이터 이동의 경로
              </span>
              로 해석합니다.  
              연산 경계마다 중간값을 HBM에 기록하는 구조를 의심하고,
              가능한 오래 데이터를 온칩에 머물게 하며
              전체 실행을 하나의 연속된 흐름으로 다시 설계합니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/theory"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-emerald-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-emerald-500 transition-all active:scale-95"
              >
                메모리 최적화 이론 보기 <ArrowRight size={18} />
              </Link>
              <Link
                to="/pipeline"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                실행 계획 흐름 탐색
              </Link>
            </div>
          </section>

          {/* NARRATIVE */}
          <section id="narrative" className="space-y-8">
            <div className="flex items-center gap-2 text-emerald-400 font-black uppercase tracking-widest text-xs">
              <Waypoints size={16} /> Physical Reality
            </div>

            <h2 className="text-4xl font-black tracking-tight text-white">
              Operator의 경계에서,
              <br />
              Dataflow의 연속성으로
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {[
                {
                  k: "Residency",
                  t: "데이터의 체류 시간",
                  sub: "On-Chip Lifetime",
                  icon: <Sparkles size={18} />,
                  p1: "중간 결과를 생성할 때마다 외부 메모리에 기록하는 구조는 현대 가속기에서 가장 비싼 비용이 됩니다.",
                  p2: "AICF는 값을 가능한 오래 레지스터, shared memory, 온칩 경로에 유지하는 방향으로 실행을 재조직합니다.",
                },
                {
                  k: "Boundary",
                  t: "연산 경계의 해체",
                  sub: "Boundary Elimination",
                  icon: <Layers size={18} />,
                  p1: "기존 operator 단위 실행은 논리적으로는 명확하지만, 물리적으로는 불필요한 write/read 사이클을 반복시킵니다.",
                  p2: "우리는 독립된 연산이 아니라, 연결된 데이터 흐름을 하나의 실행 단위로 보려 합니다.",
                },
                {
                  k: "Constraint",
                  t: "물리적 제약 기반 계획",
                  sub: "Traffic-Aware Planning",
                  icon: <ShieldCheck size={18} />,
                  p1: "컴파일러는 단순한 수식 변환이 아니라, 대역폭·레이턴시·온칩 용량을 고려하는 물리적 계획기여야 합니다.",
                  p2: "최적화의 한계는 의미 손실이 아니라, 하드웨어의 resident capacity와 traffic budget에서 결정됩니다.",
                },
              ].map((s) => (
                <div
                  key={s.k}
                  className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl hover:border-emerald-500/40 transition group"
                >
                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] uppercase tracking-[0.25em] font-black">
                    {s.icon} {s.k}
                  </div>
                  <div className="mt-4">
                    <div className="text-emerald-100 font-black text-xl tracking-tight leading-tight uppercase">
                      {s.t}
                    </div>
                    <div className="text-emerald-500/60 font-mono text-[11px] font-bold uppercase tracking-wider mt-1">
                      {s.sub}
                    </div>
                  </div>
                  <p className="mt-6 text-slate-400 leading-relaxed text-[15px]">
                    {s.p1}
                  </p>
                  <p className="mt-3 text-slate-500 leading-relaxed text-[14px] italic">
                    {s.p2}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* EXECUTION STRATEGY */}
          <section className="space-y-10 py-10">
            <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
              <Boxes size={16} /> Memory-Oriented Execution Strategy
            </div>

            <div className="flex flex-col lg:flex-row gap-12 items-start">
              <div className="lg:w-1/2 space-y-6">
                <h2 className="text-4xl font-black tracking-tight text-white leading-tight">
                  커널을 나누는 대신, <br />
                  <span className="text-emerald-400">데이터가 흐르는 경로</span>를 컴파일하다
                </h2>

                <p className="text-slate-400 text-lg leading-relaxed">
                  AICF의 Memory Optimization은 단순한 fusion 체크리스트가 아닙니다.
                  <br />
                  어떤 값을 언제 생성하고, 얼마나 오래 유지하고,
                  어느 시점에만 외부 메모리로 내보낼지를 결정하는
                  <span className="text-slate-100 font-bold">
                    {" "}
                    dataflow planner
                  </span>
                  입니다.
                </p>

                <div className="grid grid-cols-2 gap-4 pt-4">
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="text-emerald-400 font-black text-2xl mb-1">
                      Minimized
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-tighter">
                      HBM Traffic
                    </div>
                  </div>
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="text-emerald-400 font-black text-2xl mb-1">
                      Fused
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-tighter">
                      Execution Path
                    </div>
                  </div>
                </div>
              </div>

              {/* Visual architecture card */}
              <div className="lg:w-1/2 w-full bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8 relative overflow-hidden group">
                <div className="absolute inset-0 bg-emerald-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="space-y-6 relative">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-4 text-xs font-mono text-slate-500 uppercase tracking-widest">
                    <span>Memory Execution Workflow</span>
                    <span className="text-emerald-500/50">Traffic Planner v1.0</span>
                  </div>

                  <div className="space-y-4">
                    {[
                      {
                        label: "Intermediate Lifetime Analysis",
                        detail: "Which values must survive across stages?",
                        color: "bg-blue-500",
                      },
                      {
                        label: "Residency-Aware Fusion Planning",
                        detail: "Can this path remain on-chip?",
                        color: "bg-indigo-500",
                      },
                      {
                        label: "Traffic-Bounded Launch Strategy",
                        detail: "Write back only when necessary",
                        color: "bg-emerald-500",
                      },
                    ].map((step, i) => (
                      <div key={i} className="flex items-center gap-4">
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

                  <Link
                    to="/pipeline"
                    className="mt-8 flex items-center justify-center w-full py-4 rounded-xl border border-emerald-500/30 text-emerald-400 font-bold text-xs uppercase tracking-widest hover:bg-emerald-500 hover:text-[#0b1120] transition-all"
                  >
                    메모리 실행 계획 상세 보기
                  </Link>
                </div>
              </div>
            </div>
          </section>

          {/* FINAL STATEMENT */}
          <section className="bg-[#111827] border border-slate-800 rounded-[2.5rem] p-10 sm:p-12 relative overflow-hidden">
            <div className="absolute -bottom-10 -right-10 text-[120px] font-black text-emerald-500/5 pointer-events-none uppercase tracking-tighter">
              Dataflow
            </div>

            <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <Workflow size={16} /> AICF Position
            </div>

            <h3 className="mt-5 text-3xl sm:text-4xl font-black tracking-tight text-white leading-tight">
              Memory Optimization은
              <br />
              “어떻게 계산할까”가 아니라
              <br />
              “어디에 값을 남길까”의 문제다
            </h3>

            <p className="mt-6 max-w-3xl text-slate-400 text-lg leading-relaxed">
              성능은 더 빠른 ALU만으로 오르지 않습니다.  
              값이 kernel 사이를 왕복하지 않도록 만들고,
              operator의 경계를 물리적 실행 경계와 분리하는 순간,
              비로소 현대 GPU 최적화가 시작됩니다.
            </p>

            <div className="mt-8 flex flex-wrap gap-3">
              {[
                "On-Chip Residency",
                "Boundary Elimination",
                "Dataflow Scheduling",
                "Traffic-Aware Planning",
              ].map((tag) => (
                <span
                  key={tag}
                  className="px-4 py-2 rounded-full bg-emerald-500/5 border border-emerald-500/20 text-[11px] font-black uppercase tracking-widest text-emerald-400/80"
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