import React, { useState } from 'react';
import { 
  GitMerge, Cpu, Code2, Zap, ArrowRight, 
  Settings2, Activity, ShieldCheck, Box
} from 'lucide-react';
import AppSidebar from '../components/AppSidebar';

export default function PipelinePage() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      title: "Graph Ingestion",
      icon: <Box size={20} />,
      desc: "수학적 정의(Theory)를 바탕으로 상위 프레임워크의 그래프를 분석합니다.",
      detail: "ONNX/PyTorch IR을 AICF 전용 Semantic Graph로 변환하며, 각 노드에 이론적 제약 사항을 바인딩합니다."
    },
    {
      title: "Optimization Pass",
      icon: <Zap size={20} />,
      desc: "Ops Explorer의 전략을 적용하여 커널 퓨전 및 그래프 변환을 수행합니다.",
      detail: "Dead Code Elimination, Operator Fusion, Constant Folding 등이 이 단계에서 실행됩니다."
    },
    {
      title: "Lowering & CodeGen",
      icon: <Code2 size={20} />,
      desc: "최종 하드웨어 아키텍처에 최적화된 로우레벨 코드를 생성합니다.",
      detail: "선택된 Variant에 따라 Triton 혹은 CUDA 코드가 생성되며, 메모리 레이아웃이 확정됩니다."
    }
  ];

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased font-sans">
      <AppSidebar activeMenu="pipeline" />
      
      <main className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-12 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
        {/* Hero Section */}
        <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-16 shadow-2xl relative overflow-hidden">
          <div className="absolute -top-10 -right-10 text-[140px] font-black text-blue-500/5 pointer-events-none uppercase tracking-tighter">Pipeline</div>
          <div className="flex items-center gap-2 text-blue-500 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
            <Settings2 size={16} /> AICF Execution Flow
          </div>
          <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
            From Theory to <br/><span className="text-blue-500">Optimized Kernel</span>
          </h1>
          <p className="mt-8 text-slate-400 text-lg leading-relaxed max-w-3xl">
            AICF 컴파일러 파이프라인은 수학적 추상화를 물리적 성능으로 변환하는 일련의 과정을 관리합니다. 
            이 페이지에서는 각 단계에서 발생하는 데이터 변환과 최적화 알고리즘을 시각화합니다.
          </p>
        </section>

        {/* Interactive Pipeline Steps */}
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-emerald-400">
            <Activity size={24} />
            <h2 className="text-2xl font-black uppercase tracking-tight text-white">Execution Steps</h2>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {steps.map((step, idx) => (
              <div 
                key={idx}
                onClick={() => setActiveStep(idx)}
                className={`cursor-pointer p-8 rounded-[2rem] border transition-all duration-300 ${
                  activeStep === idx 
                  ? 'bg-blue-600/10 border-blue-500 shadow-[0_0_30px_rgba(59,130,246,0.1)]' 
                  : 'bg-[#1e293b] border-slate-800 hover:border-slate-700'
                }`}
              >
                <div className={`w-12 h-12 rounded-2xl flex items-center justify-center mb-6 ${
                  activeStep === idx ? 'bg-blue-600 text-white' : 'bg-slate-800 text-slate-400'
                }`}>
                  {step.icon}
                </div>
                <h3 className="text-xl font-black text-white uppercase mb-2">{step.title}</h3>
                <p className="text-slate-400 text-sm leading-relaxed">{step.desc}</p>
                {activeStep === idx && (
                  <div className="mt-6 pt-6 border-t border-blue-500/20 animate-in fade-in slide-in-from-top-2">
                    <p className="text-blue-300 text-sm font-medium">{step.detail}</p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </section>

        {/* Visualizer Placeholder */}
        <section className="bg-[#0b1120] border border-slate-800 rounded-[3rem] p-12 flex flex-col items-center justify-center min-h-[400px] text-center">
           <div className="bg-blue-500/10 p-6 rounded-full mb-6">
              <GitMerge size={48} className="text-blue-500 animate-pulse" />
           </div>
           <h3 className="text-2xl font-black text-white uppercase mb-4">Pipeline Visualizer</h3>
           <p className="text-slate-500 max-w-xl">
             각 단계별 텐서 그래프의 변화와 코드 생성 결과를 인터랙티브하게 보여주는 시각화 모듈이 이곳에 위치합니다. 
             (예: React-flow를 이용한 그래프 최적화 과정 시각화)
           </p>
        </section>
      </main>
    </div>
  );
}