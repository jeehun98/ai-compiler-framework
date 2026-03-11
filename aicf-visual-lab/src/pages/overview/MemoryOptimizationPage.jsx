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
        activeOpId={null}
        quickOps={["FlashAttention", "LayerNorm", "Softmax", "GEMM"]}
        version="v1.0.4 Stable"
      />

      <main className="flex-1 flex flex-col min-w-0">
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-emerald-400 tracking-tighter uppercase text-xl">
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

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-20">
          
          {/* HERO SECTION: The Core Value */}
          <section className="bg-gradient-to-br from-[#1e293b] to-[#0f172a] border border-slate-800 rounded-[3rem] p-10 md:p-16 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[180px] font-black text-emerald-500/5 pointer-events-none tracking-tighter uppercase select-none">
              Residency
            </div>

            <div className="flex items-center gap-3 text-emerald-400 font-mono text-xs uppercase tracking-[0.4em] font-black mb-8">
              <Zap size={16} className="fill-emerald-400" /> Dataflow & Traffic Control
            </div>

            <h1 className="text-5xl md:text-7xl font-black tracking-tight leading-[1.05] text-white">
              가장 비싼 연산은,
              <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
                이동하는 데이터다
              </span>
            </h1>

            <p className="mt-8 max-w-3xl text-slate-400 text-xl leading-relaxed font-light">
              AICF는 성능 병목을 단순히 연산량(FLOPs)으로 보지 않고 
              <span className="text-slate-100 font-medium italic mx-1">데이터 이동의 엔트로피</span> 
              로 정의합니다. 연산 경계마다 발생하는 HBM I/O를 억제하고, 
              온칩 체류 시간(Residency)을 극대화하여 실행 경로를 하나로 융합합니다.
            </p>

            <div className="mt-12 flex flex-wrap gap-5">
              <Link
                to="/theory"
                className="inline-flex items-center gap-3 px-8 py-5 rounded-2xl bg-emerald-600 text-white font-bold text-sm uppercase tracking-widest shadow-xl hover:bg-emerald-500 transition-all hover:-translate-y-1 active:scale-95"
              >
                메모리 최적화 이론 보기 <ArrowRight size={18} />
              </Link>
              <Link
                to="/pipeline"
                className="inline-flex items-center gap-3 px-8 py-5 rounded-2xl border border-slate-700 text-slate-300 font-bold text-sm uppercase tracking-widest hover:bg-slate-800 transition-all"
              >
                실행 계획 흐름 탐색
              </Link>
            </div>
          </section>

          {/* NARRATIVE: Physical Reality */}
          <section id="narrative" className="space-y-10">
            <div className="flex items-center gap-2 text-emerald-500/80 font-black uppercase tracking-[0.3em] text-xs">
              <Waypoints size={16} /> Engineering Reality
            </div>

            <h2 className="text-4xl md:text-5xl font-black tracking-tight text-white">
              Operator의 논리적 경계에서,
              <br />
              <span className="text-slate-500 italic">Dataflow의 물리적 연속성으로</span>
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {[
                {
                  k: "Residency",
                  t: "데이터 점유권의 확보",
                  sub: "On-Chip Lifetime",
                  icon: <Sparkles size={20} />,
                  p1: "중간 결과가 HBM으로 방출되는 즉시 시스템은 물리적 지연에 직면합니다.",
                  p2: "우리는 값을 레지스터와 SRAM 가용 범위 내에 최대한 묶어두는 Residency Window를 설계합니다.",
                },
                {
                  k: "Boundary",
                  t: "커널 경계의 증발",
                  sub: "Boundary Elimination",
                  icon: <Layers size={20} />,
                  p1: "논리적으로 분리된 연산은 불필요한 Write-after-Read 사이클을 강제합니다.",
                  p2: "AICF는 독립된 커널이 아닌, 융합된 데이터 파이프라인을 하나의 실행 단위로 정의합니다.",
                },
                {
                  k: "Constraint",
                  t: "물리적 자원 기반 계획",
                  sub: "Traffic-Aware Planning",
                  icon: <ShieldCheck size={20} />,
                  p1: "컴파일러는 수식 변환기가 아니라, 하드웨어 예산을 계산하는 자원 계획기여야 합니다.",
                  p2: "최적화의 한계는 수학적 논리가 아닌, 칩의 실질적인 Traffic Budget에서 결정됩니다.",
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

          {/* EXECUTION STRATEGY: Dataflow Planning */}
          <section className="space-y-12 py-10">
            <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
              <Boxes size={16} /> Memory-Centric Execution Architecture
            </div>

            <div className="flex flex-col lg:flex-row gap-16 items-center">
              <div className="lg:w-1/2 space-y-8">
                <h2 className="text-4xl md:text-5xl font-black tracking-tight text-white leading-[1.1]">
                  커널을 나누는 대신, <br />
                  <span className="text-emerald-400">데이터의 경로</span>를 컴파일하다
                </h2>

                <p className="text-slate-400 text-xl leading-relaxed font-light">
                  AICF의 메모리 최적화는 단순한 퓨전 체크리스트가 아닙니다.
                  어떤 값을 언제 생성하고 얼마나 유지할지를 결정하는 
                  <span className="text-slate-100 font-bold px-1 underline decoration-emerald-500/50 underline-offset-4">
                    Dataflow Planner
                  </span> 
                  가 성능의 핵심입니다.
                </p>

                <div className="grid grid-cols-2 gap-6 pt-4">
                  <div className="p-6 rounded-[2rem] bg-slate-800/30 border border-slate-700 group hover:border-emerald-500/30 transition-colors">
                    <div className="text-emerald-400 font-black text-3xl mb-1">
                      Reduced
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-[0.2em]">
                      HBM Traffic Entropy
                    </div>
                  </div>
                  <div className="p-6 rounded-[2rem] bg-slate-800/30 border border-slate-700 group hover:border-blue-500/30 transition-colors">
                    <div className="text-blue-400 font-black text-3xl mb-1">
                      Fused
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-[0.2em]">
                      Physical Pipeline
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
                    <span className="text-emerald-500 font-bold animate-pulse">Traffic Planner Active</span>
                  </div>

                  <div className="space-y-6">
                    {[
                      {
                        label: "Tiling & Residency Window Analysis",
                        detail: "SRAM 가용 용량 대비 최적 타일 크기 결정",
                        color: "bg-blue-500",
                      },
                      {
                        label: "Single-Pass Operator Fusion",
                        detail: "결합 법칙을 이용한 누적 연산 스트리밍 정의",
                        color: "bg-indigo-500",
                      },
                      {
                        label: "IO-Optimal Kernel Generation",
                        detail: "필요 시점에만 HBM에 쓰기를 허용하는 코드 생성",
                        color: "bg-emerald-500",
                      },
                    ].map((step, i) => (
                      <div key={i} className="flex items-center gap-6 group/item">
                        <div className={`w-3 h-14 ${step.color} rounded-full shadow-lg shadow-${step.color.split('-')[1]}-500/20`} />
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
                    to="/pipeline"
                    className="mt-10 flex items-center justify-center w-full py-5 rounded-2xl bg-slate-900 border border-emerald-500/20 text-emerald-400 font-black text-sm uppercase tracking-[0.2em] hover:bg-emerald-500 hover:text-[#0b1120] transition-all shadow-inner"
                  >
                    메모리 실행 계획 상세 보기
                  </Link>
                </div>
              </div>
            </div>
          </section>

          {/* FINAL STATEMENT: The Philosophy */}
          <section className="bg-emerald-950/20 border border-emerald-500/20 rounded-[3rem] p-12 md:p-20 relative overflow-hidden text-center">
            <div className="absolute inset-0 bg-[url('https://grainy-gradients.vercel.app/noise.svg')] opacity-20 pointer-events-none"></div>
            
            <div className="flex justify-center mb-8">
              <div className="px-5 py-2 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 font-mono text-[10px] uppercase tracking-[0.4em] font-black">
                AICF Final Position
              </div>
            </div>

            <h3 className="text-4xl md:text-6xl font-black tracking-tighter text-white leading-[1.1]">
              최적화의 본질은 계산의 속도가 아니라,
              <br />
              <span className="italic text-emerald-400">데이터의 침묵(Silence)</span>
              에 있습니다.
            </h3>

            <p className="mt-10 max-w-2xl mx-auto text-slate-400 text-lg leading-relaxed font-light">
              값이 커널 사이를 왕복하며 에너지를 낭비하지 않도록, 
              Operator의 논리적 경계를 물리적 실행 경로와 완전히 일치시키는 순간, 
              비로소 진정한 가속이 시작됩니다.
            </p>

            <div className="mt-12 flex flex-wrap justify-center gap-4">
              {[
                "On-Chip Residency",
                "Boundary Elimination",
                "Dataflow Scheduling",
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