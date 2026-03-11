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
          <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-12 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[160px] font-black text-blue-500/5 pointer-events-none tracking-tighter uppercase">
              Compute
            </div>

            <div className="flex items-center gap-2 text-blue-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <Cpu size={16} /> Compute Optimization & Semantic Reduction
            </div>

            <h1 className="mt-6 text-5xl sm:text-6xl font-black tracking-tight leading-[1.1] text-white">
              같은 의미를
              <br />
              더 적은 계산으로 실행하다
            </h1>

            <p className="mt-6 max-w-3xl text-slate-400 text-lg leading-relaxed">
              AICF는 최적화를 단순한 속도 향상이 아니라,
              <span className="text-slate-100 font-semibold italic">
                {" "}
                의미를 유지한 채 계산량을 줄이는 문제
              </span>
              로 봅니다. 핵심은 어떤 계산이 본질적으로 필요하고 어떤 계산이
              대체 가능하거나 생략 가능한지를 수학적 제약 아래에서 판단하는
              것입니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/compute/theory"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-blue-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-blue-500 transition-all active:scale-95"
              >
                Theory Index 보기 <ArrowRight size={18} />
              </Link>
              <Link
                to="/compute/ops"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                연산 리포트 탐색
              </Link>
            </div>
          </section>

          <section id="narrative" className="space-y-8">
            <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
              <Waypoints size={16} /> Semantic Compute Model
            </div>

            <h2 className="text-4xl font-black tracking-tight text-white">
              연산의 나열에서,
              <br />
              의미 기반 축약으로
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {[
                {
                  k: "Definition",
                  t: "의미를 먼저 정의한다",
                  sub: "Semantic Contract",
                  icon: <Sparkles size={18} />,
                  p1: "모든 최적화는 먼저 무엇이 반드시 보존되어야 하는가에서 시작합니다. AICF는 각 연산을 단순한 코드 조각이 아니라 의미 계약으로 해석합니다.",
                  p2: "이 계약이 있어야만 계산 삭제, 재배열, 근사화가 허용 가능한지 판단할 수 있습니다.",
                },
                {
                  k: "Equivalence",
                  t: "동일한 결과를 더 작은 계산으로",
                  sub: "Equivalent Reduction",
                  icon: <Layers size={18} />,
                  p1: "서로 다른 계산 경로라도 동일한 의미 조건을 만족하면 같은 연산으로 볼 수 있습니다. 중요한 것은 수식의 모양이 아니라 결과가 지키는 구조입니다.",
                  p2: "AICF는 이 지점을 이용해 더 작고 단순한 계산 형태를 탐색합니다.",
                },
                {
                  k: "Boundary",
                  t: "허용 가능한 변형의 경계",
                  sub: "Approximation Boundary",
                  icon: <ShieldCheck size={18} />,
                  p1: "모든 축약이 허용되지는 않습니다. 계산 감소는 의미 손실이 시작되는 지점 전까지만 유효합니다.",
                  p2: "AICF는 수치 오차 자체보다 의미적 손실의 발생 여부를 최적화의 경계로 삼습니다.",
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

          <section className="space-y-10 py-10">
            <div className="flex items-center gap-2 text-emerald-400 font-black uppercase tracking-widest text-xs">
              <Boxes size={16} /> Compute-Oriented Execution Strategy
            </div>

            <div className="flex flex-col lg:flex-row gap-12 items-start">
              <div className="lg:w-1/2 space-y-6">
                <h2 className="text-4xl font-black tracking-tight text-white leading-tight">
                  코드를 생성하는 것을 넘어,
                  <br />
                  <span className="text-blue-400">의미적으로 가장 경제적인 실행</span>을 선택하다
                </h2>

                <p className="text-slate-400 text-lg leading-relaxed">
                  AICF는 모델을 곧바로 커널 호출로 번역하지 않습니다.
                  먼저 어떤 계산이 의미적으로 필수인지 분석하고, 그 다음
                  동일한 의미를 더 적은 비용으로 실현할 수 있는
                  <span className="text-slate-100 font-bold"> kernel variant</span>
                  와 실행 경로를 선택합니다.
                </p>

                <div className="grid grid-cols-2 gap-4 pt-4">
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="text-blue-400 font-black text-2xl mb-1">
                      Reduced
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-tighter">
                      Compute Cost
                    </div>
                  </div>
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="text-blue-400 font-black text-2xl mb-1">
                      Adaptive
                    </div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-tighter">
                      Variant Selection
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:w-1/2 w-full bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8 relative overflow-hidden group">
                <div className="absolute inset-0 bg-blue-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="space-y-6 relative">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-4 text-xs font-mono text-slate-500 uppercase tracking-widest">
                    <span>Compute Execution Workflow</span>
                    <span className="text-blue-500/50">Semantic Planner v1.0</span>
                  </div>

                  <div className="space-y-4">
                    {[
                      {
                        label: "Semantic Constraint Analysis",
                        detail: "What must be preserved?",
                        color: "bg-blue-500",
                      },
                      {
                        label: "Equivalent Compute Path Search",
                        detail: "Which form is cheaper but valid?",
                        color: "bg-indigo-500",
                      },
                      {
                        label: "Kernel Variant Finalization",
                        detail: "Bind execution to the selected path",
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
                    실행 계획 생성 과정 상세 보기
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