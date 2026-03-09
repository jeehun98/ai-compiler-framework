import React, { useState } from 'react';
import { 
  GitMerge, Cpu, Code2, Zap, ArrowRight, 
  Settings2, Activity, ShieldCheck, Box
} from 'lucide-react';
import ComputeSidebar from '../../../components/ComputeSidebar.jsx';

export default function PipelinePage() {
  const [activeStep, setActiveStep] = useState(0);

  const steps = [
    {
      title: "Semantic Ingestion",
      icon: <Box size={20} />,
      desc: "수학적 정의(Theory)를 바탕으로 상위 프레임워크의 그래프를 분석합니다.",
      detail: "ONNX/PyTorch IR을 AICF 전용 Semantic Graph로 변환하며, 각 노드에 이론적 제약 사항과 수학적 불변성(Invariants)을 바인딩합니다."
    },
    {
      title: "Optimization Pass",
      icon: <Zap size={20} />,
      desc: "비트마스크 패턴 매칭을 통해 최적화 기회를 식별하고 그래프를 변환합니다.",
      detail: "Dead Code Elimination뿐만 아니라, 비트마스크 매칭을 통한 Operator Fusion 및 메모리 액세스 최적화 전략이 결정됩니다."
    },
    {
      title: "Plan Concretization", // Lowering & CodeGen에서 변경
      icon: <Code2 size={20} />,
      desc: "하드웨어 타겟에 맞춰 실체화된 실행 계획(Executable Plan)을 생성합니다.",
      detail: "선택된 Variant에 따라 Launch Config, Memory Layout(Stride/Packing)이 확정되며, 최종 Executable Plan 데이터 구조가 빌드됩니다."
    }
  ];

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased font-sans">
      <ComputeSidebar activeMenu="pipeline" />
      
      <main className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-12 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
        {/* Hero Section */}
        <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-16 shadow-2xl relative overflow-hidden">
          <div className="absolute -top-10 -right-10 text-[140px] font-black text-blue-500/5 pointer-events-none uppercase tracking-tighter">Pipeline</div>
          <div className="flex items-center gap-2 text-blue-500 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
            <Settings2 size={16} /> AICF Execution Strategy
          </div>
          <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
            From Theory to <br/><span className="text-blue-500">Executable Plan</span>
          </h1>
          <p className="mt-8 text-slate-400 text-lg leading-relaxed max-w-3xl">
            AICF 컴파일러는 단순한 코드 번역기를 넘어, 수학적 추상화를 물리적 성능으로 변환하는 <strong>전략적 기획자(Planner)</strong>입니다. 
            패턴 매칭을 통해 검증된 커널을 조합하고 최적의 실행 경로를 결정하는 과정을 시각화합니다.
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
            <h3 className="text-2xl font-black text-white uppercase mb-4">Plan Orchestrator</h3>
            <p className="text-slate-500 max-w-xl">
              각 단계별 텐서 그래프의 비트마스크 매칭 결과와 최종 결정된 <strong>Execution Plan(Launch Config, Kernel ID)</strong>을 인터랙티브하게 시각화합니다. 
              (React-flow를 통해 하드웨어 자원 할당 계획을 출력합니다)
            </p>
        </section>
      </main>
    </div>
  );
}