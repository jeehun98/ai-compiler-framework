import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Cpu,
  ShieldCheck,
  Layers,
  ArrowRight,
  Boxes,
  Sparkles,
  Waypoints,
  Menu,
} from "lucide-react";
import AppSidebar from "../../components/layout/ComputeSidebar.jsx";

export default function ComputeOptimizationPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <AppSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeOpId={null}
        quickOps={["GEMM", "Softmax", "LayerNorm", "AdamStep"]}
        version="v1.0.4 Stable"
      />

      <main className="flex-1 flex flex-col min-w-0">
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-blue-400 tracking-tighter uppercase">
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
          {/* Hero Section */}
          <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-12 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[160px] font-black text-blue-500/5 pointer-events-none tracking-tighter uppercase">
              Compute
            </div>

            <div className="flex items-center gap-2 text-blue-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <Cpu size={16} /> Invariant-Preserving Runtime Optimization
            </div>

            <h1 className="mt-6 text-5xl sm:text-6xl font-black tracking-tight leading-[1.1] text-white">
              불변적 성질 안에서
              <br />
              최적의 실행 경로를 선택하다
            </h1>

            <p className="mt-6 max-w-3xl text-slate-400 text-lg leading-relaxed">
              AICF는 최적화를 단순한 코드 변환이 아니라, 
              <span className="text-slate-100 font-semibold italic">
                {" "}
                연산의 불변성(Invariants)을 유지하며 런타임 비용을 최소화하는 결정 문제
              </span>
              로 정의합니다. 핵심은 수학적 제약 조건 아래에서 하드웨어와 입력 상태에 맞는 가장 경제적인 연산 경로를 동적으로 선택하는 것입니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/compute/theory"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-blue-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-blue-500 transition-all active:scale-95"
              >
                Optimization Theory 보기 <ArrowRight size={18} />
              </Link>
              <Link
                to="/compute/ops"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                Runtime Ops 탐색
              </Link>
            </div>
          </section>

          {/* Three Pillars Section */}
          <section id="narrative" className="space-y-8">
            <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
              <Waypoints size={16} /> Semantic Compute Model
            </div>

            <h2 className="text-4xl font-black tracking-tight text-white">
              고정된 연산 그래프에서,
              <br />
              적응형 실행 구조로
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {[
                {
                  k: "Invariants",
                  t: "보존되어야 할 본질의 정의",
                  sub: "Invariant Preservation",
                  icon: <Sparkles size={18} />,
                  p1: "모든 최적화는 '무엇이 절대 변하지 않아야 하는가'에서 시작합니다. AICF는 연산을 수식이 아닌, 반드시 지켜야 할 수학적 불변성(Invariants)의 집합으로 해석합니다.",
                  p2: "이 제약 조건이 명확할 때만, 비로소 안전한 연산 축약과 경로 변경이 가능해집니다.",
                },
                {
                  k: "Adapt성",
                  t: "상황에 따른 최적 경로 탐색",
                  sub: "Runtime Adaptive Path",
                  icon: <Layers size={18} />,
                  p1: "동일한 결과에 도달하는 경로는 다양합니다. AICF는 런타임 입력 크기, 하드웨어 가용 자원에 따라 불변성을 해치지 않는 가장 가벼운 실행 경로를 탐색합니다.",
                  p2: "정적 컴파일 타임에 결정할 수 없는 실행 효율을 런타임 가드(Guard)를 통해 실현합니다.",
                },
                {
                  k: "Boundary",
                  t: "수치적 허용 오차의 설계",
                  sub: "Precision Boundary",
                  icon: <ShieldCheck size={18} />,
                  p1: "모든 계산이 완벽할 필요는 없습니다. AICF는 의미적 손실이 발생하기 직전의 경계를 파악하여, 허용 가능한 범위 내에서 과감한 연산 최적화를 수행합니다.",
                  p2: "수치적 오차보다 모델의 최종 목적지에 도달하는 '의미적 정확도'를 경계로 삼습니다.",
                },
              ].map((s) => (
                <div
                  key={s.k}
                  className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl hover:border-blue-500/40 transition group"
                >
                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] uppercase tracking-[0.25em] font-black">
                    {s.icon} {s.k}
                  </div>
                  <div className="mt-4">
                    <div className="text-blue-100 font-black text-xl tracking-tight leading-tight uppercase">
                      {s.t}
                    </div>
                    <div className="text-blue-500/60 font-mono text-[11px] font-bold uppercase tracking-wider mt-1">
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

          {/* Execution Strategy Section */}
          <section className="space-y-10 py-10">
            <div className="flex items-center gap-2 text-emerald-400 font-black uppercase tracking-widest text-xs">
              <Boxes size={16} /> Compute-Oriented Execution Strategy
            </div>

            <div className="flex flex-col lg:flex-row gap-12 items-start">
              <div className="lg:w-1/2 space-y-6">
                <h2 className="text-4xl font-black tracking-tight text-white leading-tight">
                  단순한 코드 생성을 넘어,
                  <br />
                  <span className="text-blue-400">런타임에 최적의 파라미터</span>를 바인딩하다
                </h2>

                <p className="text-slate-400 text-lg leading-relaxed">
                  AICF의 Compute 엔진은 고정된 커널을 호출하지 않습니다.
                  분석된 불변성을 바탕으로 현시점에서 가장 효율적인 
                  <span className="text-slate-100 font-bold"> Kernel Variant</span>와 
                  타일 크기, 스케줄링 파라미터를 결정하여 실행 단계에 주입합니다.
                </p>

                <div className="grid grid-cols-2 gap-4 pt-4">
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="text-blue-400 font-black text-2xl mb-1">
                      Adaptive
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-tighter">
                      Runtime Tuning
                    </div>
                  </div>
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="text-blue-400 font-black text-2xl mb-1">
                      Optimal
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-tighter">
                      Path Selection
                    </div>
                  </div>
                </div>
              </div>

              {/* Workflow Visualizer Area */}
              <div className="lg:w-1/2 w-full bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8 relative overflow-hidden group">
                <div className="absolute inset-0 bg-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="space-y-6 relative">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-4 text-xs font-mono text-slate-500 uppercase tracking-widest">
                    <span>Compute Optimization Pipeline</span>
                    <span className="text-blue-500/50">Runtime Guard v1.0</span>
                  </div>

                  <div className="space-y-4">
                    {[
                      {
                        label: "Invariant Analysis",
                        detail: "고정되어야 할 수학적 성질 파악",
                        color: "bg-blue-500",
                      },
                      {
                        label: "Runtime Path Search",
                        detail: "현재 상태에서 최적의 연산 경로 탐색",
                        color: "bg-indigo-500",
                      },
                      {
                        label: "Parameter Binding",
                        detail: "선택된 경로에 최적의 파라미터 주입",
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
                    to="/compute/pipeline"
                    className="mt-8 flex items-center justify-center w-full py-4 rounded-xl border border-blue-500/30 text-blue-400 font-bold text-xs uppercase tracking-widest hover:bg-blue-500 hover:text-[#0b1120] transition-all"
                  >
                    Dynamic Execution 과정 상세 보기
                  </Link>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}