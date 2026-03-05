import React from 'react';
import { 
  BarChart2, Zap, Activity, ShieldAlert, 
  Cpu, Database, ArrowRight, Microscope, 
  FileCode, LineChart, PieChart // pieChart를 PieChart로 수정
} from 'lucide-react';
import { Link } from 'react-router-dom';
import AppSidebar from '../components/AppSidebar';

export default function KernelAnalysisPage() {
  // 전반적인 커널 라이브러리 상태 통계
  const globalMetrics = [
    { label: "Total Kernels", value: "128", unit: "Variants", icon: <Database size={18} className="text-blue-400"/> },
    { label: "Avg. Efficiency", value: "78.4", unit: "% Peak", icon: <Zap size={18} className="text-yellow-400"/> },
    { label: "Stability", value: "99.9", unit: "Pass Rate", icon: <ShieldAlert size={18} className="text-emerald-400"/> },
    { label: "Analysed Ops", value: "24", unit: "Operators", icon: <Activity size={18} className="text-purple-400"/> }
  ];

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased font-sans">
      <AppSidebar activeMenu="ops" />

      <main className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-12 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
        
        {/* 1. HERO: Analysis Philosophy (범용적인 내용으로 수정) */}
        <header className="bg-[#1e293b] border border-slate-800 rounded-[3rem] p-10 sm:p-14 shadow-2xl relative overflow-hidden">
          <div className="absolute -top-10 -right-10 text-[140px] font-black text-emerald-500/5 pointer-events-none uppercase tracking-tighter">Analysis</div>
          <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
            <Microscope size={16} /> Kernel Understanding Pipeline
          </div>
          <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
            커널을 코드가 아닌 <br/><span className="text-emerald-400">실험의 대상</span>으로
          </h1>
          <p className="mt-8 text-slate-400 text-lg leading-relaxed max-w-3xl">
            AICF Kernel Analysis System은 단순한 지표 수집을 넘어 <strong>지표의 원인을 해석</strong>합니다.
            정적 코드 분석과 동적 프로파일링 데이터를 결합하여, 각 커널이 하드웨어 한계치에 도달하지 못하는 병목 패턴을 식별합니다.
          </p>
        </header>

        {/* 2. Global Status Cards - h-40 고정으로 레이아웃 붕괴 방지 */}
        <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {globalMetrics.map((m, i) => (
            <div key={i} className="bg-[#1e293b] border border-slate-800 p-6 rounded-[2rem] shadow-xl flex flex-col justify-between h-40 hover:border-slate-700 transition-all">
              <div className="flex justify-between items-start">
                <div className="p-3 bg-slate-800/50 rounded-2xl border border-slate-700">{m.icon}</div>
                <div className="text-slate-500 font-mono text-[10px] uppercase font-black tracking-widest">{m.label}</div>
              </div>
              <div>
                <div className="text-3xl font-black text-white leading-none mb-1">{m.value}</div>
                <div className="text-slate-500 text-[10px] font-mono uppercase tracking-widest">{m.unit}</div>
              </div>
            </div>
          ))}
        </section>

        {/* 3. 분석 아키텍처 (Static vs Dynamic) */}
        <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Static Analyzer Section */}
          <div className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 space-y-6">
            <div className="flex items-center gap-3">
              <FileCode className="text-blue-500" size={24} />
              <h3 className="text-xl font-black text-white uppercase tracking-tight">Static Kernel Analyzer</h3>
            </div>
            <p className="text-slate-500 text-sm leading-relaxed">
              실행 전, CUDA 커널 소스 코드의 구조를 분석하여 <strong>Memory Stride, Thread Mapping</strong> 등을 추론합니다.
              이를 통해 Coalescing 가능성과 잠재적 Occupancy 제한 요소를 사전에 식별합니다.
            </p>
            <div className="p-5 bg-slate-900/50 rounded-2xl border border-slate-800 font-mono text-[11px] text-slate-500 space-y-2">
              <p className="text-blue-400">// Automatic Pattern Recognition</p>
              <p>✔ Thread-Data Mapping Analysis ... <span className="text-emerald-500">Done</span></p>
              <p>✔ Arithmetic Intensity Estimation ... <span className="text-emerald-400">High</span></p>
              <p>✔ Resource Footprint Tracking ... <span className="text-yellow-500">Alert</span></p>
            </div>
          </div>

          {/* Dynamic Profiler Section */}
          <div className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 space-y-6">
            <div className="flex items-center gap-3">
              <PieChart className="text-emerald-500" size={24} />
              <h3 className="text-xl font-black text-white uppercase tracking-tight">Dynamic Profiling Layer</h3>
            </div>
            <p className="text-slate-500 text-sm leading-relaxed">
              하드웨어 메트릭을 수집하여 실제 런타임 성능을 측정합니다.
              <strong>Roofline Model</strong>을 통해 커널이 Compute-Bound인지 Memory-Bound인지 판별합니다.
            </p>
            <div className="flex items-end justify-center gap-2 h-24 pt-4">
              {[30, 60, 45, 90, 70, 50, 80].map((h, i) => (
                <div key={i} style={{ height: `${h}%` }} className="w-2 bg-slate-800 rounded-t-sm" />
              ))}
            </div>
          </div>
        </section>

        {/* 4. Optimization Strategy */}
        <section className="bg-emerald-500/5 border border-emerald-500/20 rounded-[3rem] p-10 sm:p-14">
          <div className="flex items-center gap-2 text-emerald-400 font-black text-xs uppercase tracking-[0.3em] mb-8">
            <Zap size={16} /> Optimization Suggestion Engine
          </div>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-10">
            <div>
              <h4 className="text-white font-bold mb-3 uppercase tracking-tighter">1. Performance Tracing</h4>
              <p className="text-slate-500 text-sm leading-relaxed">커널 버전별 성능 변화를 추적하고 최적화 효과를 정량적으로 검증합니다.</p>
            </div>
            <div>
              <h4 className="text-white font-bold mb-3 uppercase tracking-tighter">2. Bottleneck Detection</h4>
              <p className="text-slate-500 text-sm leading-relaxed">메모리 대역폭과 연산 사용률의 상관관계를 분석하여 병목 지점을 제안합니다.</p>
            </div>
            <div>
              <h4 className="text-white font-bold mb-3 uppercase tracking-tighter">3. Adaptive Selection</h4>
              <p className="text-slate-500 text-sm leading-relaxed">분석된 데이터를 바탕으로 AICF Planner가 최적의 커널 변체를 선택하도록 지원합니다.</p>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}