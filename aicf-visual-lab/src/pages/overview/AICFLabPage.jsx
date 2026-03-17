import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity,
  ArrowRight,
  Workflow,
  Cpu,
  HardDrive,
  FlaskConical,
  Microscope,
  Gauge,
  ShieldCheck,
  Waypoints,
  Menu,
} from "lucide-react";
import LabSidebar from "../../components/layout/LabSidebar.jsx";

export default function AICFLabPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <LabSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.0.0 Validation"
      />

      <main className="flex-1 flex flex-col min-w-0">
        {/* Mobile Header */}
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-violet-400 tracking-tighter uppercase">
              AICF Lab
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
            <div className="absolute -top-10 -right-10 text-[140px] sm:text-[160px] font-black text-violet-500/5 pointer-events-none tracking-tighter uppercase">
              Lab
            </div>

            <div className="flex items-center gap-2 text-violet-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <Activity size={16} /> Execution, Observation, Validation
            </div>

            <h1 className="mt-6 text-4xl sm:text-6xl font-black tracking-tight leading-[1.08] text-white">
              설계된 최적화가
              <br />
              <span className="text-violet-400">실제 실행에서 드러나는 공간</span>
            </h1>

            <p className="mt-6 max-w-3xl text-slate-400 text-base sm:text-lg leading-relaxed">
              AICF Lab은 Compute와 Memory에서 정의된 설계가 실제 런타임에서
              어떻게 동작하는지 관측하고 검증하는 계층입니다.
              <span className="text-slate-100 font-semibold italic">
                {" "}
                execution pipeline
              </span>
              ,
              <span className="text-slate-100 font-semibold italic">
                {" "}
                kernel behavior
              </span>
              ,
              <span className="text-slate-100 font-semibold italic">
                {" "}
                memory residency
              </span>
              를 함께 다루며, 추상적 설계를 측정 가능한 실행 현상으로 연결합니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/lab/pipeline"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-violet-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-violet-500 transition-all active:scale-95"
              >
                Execution Pipeline 보기 <ArrowRight size={18} />
              </Link>

              <Link
                to="/lab/analysis"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                Kernel Analysis 보기
              </Link>
            </div>
          </section>

          {/* Core Sections */}
          <section className="space-y-8">
            <div className="flex items-center gap-2 text-violet-400 font-black uppercase tracking-widest text-xs">
              <Waypoints size={16} /> Three Surfaces of Lab
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              {/* Pipeline */}
              <Link
                to="/lab/pipeline"
                className="group bg-[#1e293b]/60 border border-slate-800 rounded-[2rem] p-8 hover:border-blue-500/30 transition"
              >
                <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-400 mb-6">
                  <Workflow size={24} />
                </div>

                <h3 className="text-2xl font-black text-white group-hover:text-blue-400 transition-colors">
                  Pipeline
                </h3>

                <p className="mt-4 text-slate-400 leading-relaxed text-[15px]">
                  의미 보존 조건 아래에서 실제 실행 경로가 어떻게 선택되고,
                  parameter binding과 kernel realization이 어떤 흐름으로
                  이어지는지 추적합니다.
                </p>

                <ul className="mt-6 space-y-2 text-sm text-slate-500 font-mono">
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-blue-500" />
                    Invariant Check
                  </li>
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-blue-500" />
                    Path Selection
                  </li>
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-blue-500" />
                    Runtime Binding
                  </li>
                </ul>
              </Link>

              {/* Kernel Analysis */}
              <Link
                to="/lab/analysis"
                className="group bg-[#1e293b]/60 border border-slate-800 rounded-[2rem] p-8 hover:border-violet-500/30 transition"
              >
                <div className="w-12 h-12 rounded-xl bg-violet-500/10 flex items-center justify-center text-violet-400 mb-6">
                  <Microscope size={24} />
                </div>

                <h3 className="text-2xl font-black text-white group-hover:text-violet-400 transition-colors">
                  Kernel Analysis
                </h3>

                <p className="mt-4 text-slate-400 leading-relaxed text-[15px]">
                  선택된 커널이 실제 하드웨어에서 어떤 성능 특성과 병목을 보이는지
                  분석합니다. Nsight, metric, trace를 통해 실행을 해석 가능한
                  신호로 바꿉니다.
                </p>

                <ul className="mt-6 space-y-2 text-sm text-slate-500 font-mono">
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-violet-500" />
                    Throughput Signals
                  </li>
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-violet-500" />
                    Bottleneck Reading
                  </li>
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-violet-500" />
                    Variant Comparison
                  </li>
                </ul>
              </Link>

              {/* Experiments */}
              <Link
                to="/lab/experiments"
                className="group bg-[#1e293b]/60 border border-slate-800 rounded-[2rem] p-8 hover:border-emerald-500/30 transition"
              >
                <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-400 mb-6">
                  <FlaskConical size={24} />
                </div>

                <h3 className="text-2xl font-black text-white group-hover:text-emerald-400 transition-colors">
                  Experiments
                </h3>

                <p className="mt-4 text-slate-400 leading-relaxed text-[15px]">
                  Compute와 Memory에서 세운 가설을 반복 가능한 실험으로
                  구성합니다. 입력, 장치 상태, variant 조건을 바꿔가며 설계
                  의도가 실제로 성립하는지 검증합니다.
                </p>

                <ul className="mt-6 space-y-2 text-sm text-slate-500 font-mono">
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-emerald-500" />
                    Controlled Inputs
                  </li>
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-emerald-500" />
                    Repeatable Conditions
                  </li>
                  <li className="flex items-center gap-2">
                    <Gauge size={14} className="text-emerald-500" />
                    Validation Reports
                  </li>
                </ul>
              </Link>
            </div>
          </section>

          {/* Flow Section */}
          <section className="bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8 sm:p-10 relative overflow-hidden">
            <div className="text-center max-w-2xl mx-auto mb-12">
              <h2 className="text-3xl font-black text-white italic tracking-tight">
                "From Design to Measured Reality"
              </h2>
              <p className="mt-4 text-slate-500 leading-relaxed">
                Lab은 설계 문서를 보여주는 공간이 아니라, 설계가 실제 실행에서
                어떤 현상으로 나타나는지 확인하는 공간입니다.
              </p>
            </div>

            <div className="flex flex-col md:flex-row items-center justify-between gap-6 max-w-5xl mx-auto">
              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-cyan-900/30 border border-cyan-500/50 text-cyan-400 font-bold">
                  Compute
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Semantic Design
                </span>
              </div>

              <ArrowRight className="text-slate-700 hidden md:block" />

              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-emerald-900/30 border border-emerald-500/50 text-emerald-400 font-bold">
                  Memory
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Physical Structure
                </span>
              </div>

              <ArrowRight className="text-slate-700 hidden md:block" />

              <div className="flex flex-col items-center">
                <div className="px-6 py-3 rounded-full bg-violet-900/30 border border-violet-500/50 text-violet-400 font-bold">
                  Lab
                </div>
                <span className="text-[10px] text-slate-600 mt-2 font-mono italic">
                  Measured Reality
                </span>
              </div>
            </div>
          </section>

          {/* Validation Layer */}
          <section className="space-y-10 py-2">
            <div className="flex items-center gap-2 text-violet-400 font-black uppercase tracking-widest text-xs">
              <ShieldCheck size={16} /> Validation Layer
            </div>

            <div className="flex flex-col lg:flex-row gap-10 items-start">
              <div className="lg:w-1/2 space-y-6">
                <h2 className="text-4xl font-black tracking-tight text-white leading-tight">
                  실행은 결과만이 아니라,
                  <br />
                  <span className="text-violet-400">관측 가능한 구조</span>
                  로 남아야 한다
                </h2>

                <p className="text-slate-400 text-lg leading-relaxed">
                  Lab의 목적은 단순히 “돌아간다”를 확인하는 것이 아닙니다. 어떤
                  경로가 선택되었는지, 어떤 커널이 호출되었는지, 어디서 병목이
                  발생했는지, 데이터가 어디에 머물렀는지를 관측 가능한 형태로
                  드러내는 데 있습니다.
                </p>

                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 pt-2">
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="flex items-center gap-2 text-violet-400 font-black text-lg">
                      <Cpu size={18} />
                      Trace
                    </div>
                    <div className="mt-2 text-slate-500 text-xs uppercase font-bold tracking-wider">
                      Runtime Signals
                    </div>
                  </div>

                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="flex items-center gap-2 text-violet-400 font-black text-lg">
                      <Activity size={18} />
                      Analyze
                    </div>
                    <div className="mt-2 text-slate-500 text-xs uppercase font-bold tracking-wider">
                      Bottleneck Reading
                    </div>
                  </div>

                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="flex items-center gap-2 text-violet-400 font-black text-lg">
                      <HardDrive size={18} />
                      Validate
                    </div>
                    <div className="mt-2 text-slate-500 text-xs uppercase font-bold tracking-wider">
                      Residency Check
                    </div>
                  </div>
                </div>
              </div>

              <div className="lg:w-1/2 w-full bg-[#111827] border border-slate-800 rounded-[2.5rem] p-8 relative overflow-hidden group">
                <div className="absolute inset-0 bg-violet-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />

                <div className="relative space-y-6">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-4 text-xs font-mono text-slate-500 uppercase tracking-widest">
                    <span>Lab Observation Flow</span>
                    <span className="text-violet-500/60">Trace → Explain</span>
                  </div>

                  <div className="space-y-4">
                    {[
                      {
                        label: "Capture Execution",
                        detail: "실제 실행 경로와 kernel dispatch를 기록",
                        color: "bg-blue-500",
                      },
                      {
                        label: "Read Signals",
                        detail: "metric, trace, residency 정보를 해석",
                        color: "bg-violet-500",
                      },
                      {
                        label: "Validate Assumptions",
                        detail: "설계 가설과 실제 실행 결과를 대조",
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
                      to="/lab/experiments"
                      className="flex items-center justify-center w-full py-4 rounded-xl border border-violet-500/30 text-violet-400 font-bold text-xs uppercase tracking-widest hover:bg-violet-500 hover:text-[#0b1120] transition-all"
                    >
                      Experiments 자세히 보기
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