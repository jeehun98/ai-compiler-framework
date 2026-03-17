import React, { useState } from "react";
import { useParams } from "react-router-dom";
import {
  Microscope,
  FileCode,
  PieChart,
  Zap,
  Database,
  Activity,
  ShieldCheck,
  Menu,
} from "lucide-react";

import LabSidebar from "../../../components/layout/LabSidebar.jsx";
import OpsComparisonView from "../../../features/lab/OpsComparisonView.jsx";
import KernelDetailView from "../../../features/lab/KernelDetailView.jsx";
import { allAnalysisConfigs } from "../../../data/analysis/configs/index.js";

export default function LabAnalysisPage() {
  const { opId, kernelId } = useParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const opData = opId ? allAnalysisConfigs[opId] : null;
  const kernelData = kernelId
    ? opData?.variants?.find((v) => v.id === kernelId)
    : null;

  const globalMetrics = [
    {
      label: "Total Kernels",
      value: "128",
      unit: "Variants",
      icon: <Database size={18} className="text-blue-400" />,
    },
    {
      label: "Avg. Efficiency",
      value: "78.4",
      unit: "% Peak",
      icon: <Zap size={18} className="text-yellow-400" />,
    },
    {
      label: "Validation",
      value: "99.9",
      unit: "Pass Rate",
      icon: <ShieldCheck size={18} className="text-emerald-400" />,
    },
    {
      label: "Analysed Ops",
      value: "24",
      unit: "Operators",
      icon: <Activity size={18} className="text-violet-400" />,
    },
  ];

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased font-sans overflow-x-hidden">
      <LabSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.0.0 Validation"
      />

      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* Mobile Header */}
        <header className="md:hidden sticky top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-violet-400 tracking-tighter uppercase flex items-center gap-2">
              <Microscope size={18} /> AICF Lab
            </div>
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200 active:scale-95 transition-transform"
              aria-label="Open sidebar"
            >
              <Menu size={20} />
            </button>
          </div>
        </header>

        <div className="p-6 sm:p-10 space-y-12 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          {kernelId ? (
            <KernelDetailView kernelData={kernelData} />
          ) : opId ? (
            <OpsComparisonView opData={opData} />
          ) : (
            <>
              {/* Hero */}
              <section className="bg-[#1e293b] border border-slate-800 rounded-[3rem] p-10 sm:p-14 shadow-2xl relative overflow-hidden mt-4 md:mt-0">
                <div className="absolute -top-10 -right-10 text-[140px] font-black text-violet-500/5 pointer-events-none uppercase">
                  Analysis
                </div>

                <div className="flex items-center gap-2 text-violet-400 font-mono text-xs font-black uppercase mb-6 tracking-widest">
                  <Microscope size={16} /> Kernel Observation & Validation
                </div>

                <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                  커널을 코드가 아니라
                  <br />
                  <span className="text-violet-400">관측 가능한 실행 현상</span>으로
                </h1>

                <p className="mt-8 text-slate-400 text-lg leading-relaxed max-w-3xl">
                  AICF Lab Analysis는 정적 구조와 동적 실행 신호를 함께 읽어,
                  선택된 커널이 실제 하드웨어에서 어떤 병목과 효율 패턴을
                  보이는지 해석합니다. 목표는 단순한 프로파일링이 아니라,
                  설계된 최적화가 실제로 어떻게 드러나는지 검증 가능한 형태로
                  만드는 것입니다.
                </p>
              </section>

              {/* Metrics */}
              <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {globalMetrics.map((m) => (
                  <div
                    key={m.label}
                    className="bg-[#1e293b] border border-slate-800 p-6 rounded-[2rem] shadow-xl flex flex-col justify-between h-40 group hover:border-violet-500/30 transition-colors"
                  >
                    <div className="flex justify-between items-start gap-4">
                      <div className="p-3 bg-slate-800/50 rounded-2xl border border-slate-700 group-hover:bg-slate-700 transition-colors">
                        {m.icon}
                      </div>
                      <div className="text-right text-slate-500 font-mono text-[10px] uppercase font-black tracking-widest">
                        {m.label}
                      </div>
                    </div>

                    <div>
                      <div className="text-3xl font-black text-white leading-none mb-1">
                        {m.value}
                      </div>
                      <div className="text-slate-500 text-[10px] font-mono uppercase">
                        {m.unit}
                      </div>
                    </div>
                  </div>
                ))}
              </section>

              {/* Analysis Surfaces */}
              <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 space-y-6 hover:bg-[#0f172a] transition-colors">
                  <div className="flex items-center gap-3">
                    <FileCode className="text-blue-500" size={24} />
                    <h3 className="text-xl font-black text-white uppercase">
                      Static Reading
                    </h3>
                  </div>
                  <p className="text-slate-500 text-sm leading-relaxed">
                    CUDA 소스 구조와 커널 형태를 바탕으로 thread mapping, memory
                    stride, reduction shape 같은 정적 특성을 읽어냅니다. 실행
                    이전에 드러나는 구조적 한계를 먼저 해석합니다.
                  </p>
                </div>

                <div className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 space-y-6 hover:bg-[#0f172a] transition-colors">
                  <div className="flex items-center gap-3">
                    <PieChart className="text-violet-500" size={24} />
                    <h3 className="text-xl font-black text-white uppercase">
                      Dynamic Signals
                    </h3>
                  </div>
                  <p className="text-slate-500 text-sm leading-relaxed">
                    Nsight Compute와 런타임 메트릭을 통해 실제 throughput, 병목,
                    occupancy, memory traffic 패턴을 읽습니다. 실행 중 드러난
                    신호를 통해 설계 의도를 검증합니다.
                  </p>
                </div>
              </section>

              {/* Observation Flow */}
              <section className="bg-[#111827] border border-slate-800 rounded-[2.5rem] p-8 sm:p-10 relative overflow-hidden">
                <div className="flex items-center gap-2 text-violet-400 font-black uppercase tracking-widest text-xs mb-6">
                  <Activity size={16} /> Analysis Flow
                </div>

                <div className="space-y-4">
                  {[
                    {
                      title: "Read Kernel Structure",
                      desc: "연산 형태와 thread / memory 구조를 먼저 파악합니다.",
                      color: "bg-blue-500",
                    },
                    {
                      title: "Capture Runtime Signals",
                      desc: "실행 중 발생하는 throughput, traffic, stall 신호를 수집합니다.",
                      color: "bg-violet-500",
                    },
                    {
                      title: "Explain Bottlenecks",
                      desc: "정적 구조와 동적 지표를 연결해 실제 병목 원인을 해석합니다.",
                      color: "bg-emerald-500",
                    },
                  ].map((item) => (
                    <div key={item.title} className="flex items-center gap-4">
                      <div className={`w-2 h-14 ${item.color} rounded-full`} />
                      <div>
                        <div className="text-white font-bold text-sm">
                          {item.title}
                        </div>
                        <div className="text-slate-500 text-xs font-mono mt-1">
                          {item.desc}
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
}