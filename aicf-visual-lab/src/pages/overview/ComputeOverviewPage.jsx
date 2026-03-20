import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Cpu,
  ArrowRight,
  Sparkles,
  Waypoints,
  Menu,
  Shapes,
  Gauge,
  ShieldCheck,
  Boxes,
  Workflow,
  Orbit,
} from "lucide-react";
import AppSidebar from "../../components/layout/ComputeSidebar.jsx";

export default function ComputeOverviewPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <AppSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.1.0 Property View"
      />

      <main className="flex-1 flex flex-col min-w-0">
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
          <section className="bg-gradient-to-br from-[#1e293b] to-[#0f172a] border border-slate-800 rounded-[2.5rem] p-8 sm:p-12 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[140px] sm:text-[160px] font-black text-blue-500/5 pointer-events-none tracking-tighter uppercase">
              Property
            </div>

            <div className="flex items-center gap-2 text-blue-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <Cpu size={16} /> Property to Executable Form
            </div>

            <h1 className="mt-6 text-4xl sm:text-6xl font-black tracking-tight leading-[1.08] text-white">
              연산이 허용하는 성질과
              <br />
              <span className="text-blue-500">반드시 지켜야 할 조건</span>이
              계산 구조를 결정하는 공간
            </h1>

            <p className="mt-6 max-w-3xl text-slate-400 text-base sm:text-lg leading-relaxed">
              Compute 레이어는 연산을 단순한 커널 호출로 보지 않습니다.
              먼저 연산이나 구조가 어떤 변환을 허용하는지{" "}
              <span className="text-slate-100 font-semibold italic">
                Property
              </span>
              로 보고, 그 과정에서도 반드시 유지되어야 하는 의미적 조건을{" "}
              <span className="text-slate-100 font-semibold italic">
                Invariant
              </span>
              로 고정합니다. 그리고 개별 operator는 이 성질과 조건 위에서 어떤
              lowering family로 이어질 수 있는지 분석됩니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/compute/properties"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-blue-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-blue-500 transition-all active:scale-95"
              >
                Property Atlas 탐색 <ArrowRight size={18} />
              </Link>

              <Link
                to="/compute/invariants"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-purple-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-purple-500 transition-all active:scale-95"
              >
                Invariant Atlas 탐색 <ArrowRight size={18} />
              </Link>

              <Link
                to="/compute/ops"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                Ops Explorer 보기
              </Link>
            </div>
          </section>

          <section className="space-y-8">
            <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
              <Waypoints size={16} /> Core Structure of Compute
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="bg-[#1e293b]/60 border border-slate-800 rounded-[2rem] p-8 hover:border-blue-500/30 transition group">
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-400 mb-6">
                  <Shapes size={24} />
                </div>

                <h3 className="text-2xl font-black text-white">
                  Property: What Is Allowed
                </h3>

                <p className="mt-4 text-slate-400 leading-relaxed text-[15px]">
                  Property는 이 연산이나 구조가 어떤 변환을 허용하는지를
                  나타냅니다. 재배치, 분해, 병합, 타일링, 재구성 같은 변환이
                  의미 보존 아래에서 가능한지 규정합니다.
                </p>

                <ul className="mt-6 space-y-2 text-sm text-slate-500 font-mono">
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-blue-500" />
                    Order Rewritable
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-blue-500" />
                    Tile Composable
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-blue-500" />
                    Rematerializable
                  </li>
                </ul>
              </div>

              <div className="bg-[#1e293b]/60 border border-slate-800 rounded-[2rem] p-8 hover:border-purple-500/30 transition group">
                <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center text-purple-400 mb-6">
                  <ShieldCheck size={24} />
                </div>

                <h3 className="text-2xl font-black text-white">
                  Invariant: What Must Remain
                </h3>

                <p className="mt-4 text-slate-400 leading-relaxed text-[15px]">
                  Invariant는 허용된 변환 이후에도 반드시 유지되어야 하는 의미적
                  조건입니다. 출력 분포, 정렬성, 정규화 조건, 상태 일관성 같은
                  보존 조건이 여기에 해당합니다.
                </p>

                <ul className="mt-6 space-y-2 text-sm text-slate-500 font-mono">
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-purple-500" />
                    Semantic Consistency
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-purple-500" />
                    Numeric Stability
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-purple-500" />
                    Structural Preservation
                  </li>
                </ul>
              </div>

              <div className="bg-[#1e293b]/60 border border-slate-800 rounded-[2rem] p-8 hover:border-emerald-500/30 transition group">
                <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-400 mb-6">
                  <Workflow size={24} />
                </div>

                <h3 className="text-2xl font-black text-white">
                  Ops: The Concrete Profile
                </h3>

                <p className="mt-4 text-slate-400 leading-relaxed text-[15px]">
                  Ops Explorer는 개별 operator가 어떤 property profile을 가지는지,
                  어떤 invariant를 갖는지, 그리고 어떤 lowering family가
                  자연스러운지를 보여줍니다.
                </p>

                <ul className="mt-6 space-y-2 text-sm text-slate-500 font-mono">
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-emerald-500" />
                    Property Profile
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-emerald-500" />
                    Op-Specific Constraints
                  </li>
                  <li className="flex items-center gap-2">
                    <Sparkles size={14} className="text-emerald-500" />
                    Lowering Candidates
                  </li>
                </ul>
              </div>
            </div>
          </section>

          <section className="bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8 sm:p-10 relative overflow-hidden">
            <div className="text-center max-w-2xl mx-auto mb-12">
              <h2 className="text-3xl font-black text-white italic tracking-tight">
                "From Property to Runtime"
              </h2>
              <p className="mt-4 text-slate-500 leading-relaxed">
                Compute는 의미를 직접 설명하는 계층이 아니라, 허용된 변환과
                유지되어야 할 조건을 바탕으로 실제 실행 형태를 좁혀가는
                계층입니다.
              </p>
            </div>

            <div className="flex flex-col md:flex-row items-center justify-between gap-6 max-w-5xl mx-auto">
              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-blue-900/30 border border-blue-500/50 text-blue-400 font-bold">
                  Property
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Transform Permission
                </span>
              </div>

              <ArrowRight className="text-slate-700 hidden md:block" />

              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-purple-900/30 border border-purple-500/50 text-purple-400 font-bold">
                  Invariant
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Semantic Condition
                </span>
              </div>

              <ArrowRight className="text-slate-700 hidden md:block" />

              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-emerald-900/30 border border-emerald-500/50 text-emerald-400 font-bold">
                  Ops
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Operator Profile
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
            </div>
          </section>

          <section className="space-y-10 py-2">
            <div className="flex items-center gap-2 text-violet-400 font-black uppercase tracking-widest text-xs">
              <Gauge size={16} /> Runtime Selection Layer
            </div>

            <div className="flex flex-col lg:flex-row gap-10 items-start">
              <div className="lg:w-1/2 space-y-6">
                <h2 className="text-4xl font-black tracking-tight text-white leading-tight">
                  허용된 변환 공간 안에서,
                  <br />
                  <span className="text-violet-400">
                    가장 경제적인 실행 경로
                  </span>
                  를 선택하다
                </h2>

                <p className="text-slate-400 text-lg leading-relaxed">
                  Runtime은 아무 경로나 고르는 것이 아니라, Property가 허용한
                  공간과 Invariant가 요구하는 보존 조건 안에서 현재 입력 shape,
                  메모리 상태, 장치 자원에 맞는 가장 경제적인 경로를 선택합니다.
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
                      Candidate Path
                    </div>
                  </div>

                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="flex items-center gap-2 text-violet-400 font-black text-lg">
                      <Orbit size={18} />
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
                        label: "Property Match",
                        detail: "허용 가능한 변환/realization 후보를 좁힘",
                        color: "bg-blue-500",
                      },
                      {
                        label: "Invariant Check",
                        detail: "의미 보존 조건을 만족하는지 검증",
                        color: "bg-purple-500",
                      },
                      {
                        label: "Path Binding",
                        detail: "현재 shape / resource 조건에 맞게 실행 경로 선택",
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