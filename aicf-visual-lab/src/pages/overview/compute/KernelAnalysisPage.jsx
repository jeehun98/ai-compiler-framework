import React, { useState } from 'react';
import { useParams } from 'react-router-dom';
import { Microscope, FileCode, PieChart, Zap, Database, Activity, ShieldAlert, Menu } from 'lucide-react';

import ComputeSidebar from '../../../components/layout/ComputeSidebar.jsx';
import OpsComparisonView from '../../../features/analysis/OpsComparisonView';
import KernelDetailView from '../../../features/analysis/KernelDetailView';
import { allAnalysisConfigs } from '../../../data/analysis/configs';

export default function KernelAnalysisPage() {
  const { opId, kernelId } = useParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false); // 사이드바 상태 추가
  
  // 데이터 매핑
  const opData = allAnalysisConfigs[opId];
  const kernelData = opData?.variants.find(v => v.id === kernelId);

  // 전역 통계 (초기화면용)
  const globalMetrics = [
    { label: "Total Kernels", value: "128", unit: "Variants", icon: <Database size={18} className="text-blue-400"/> },
    { label: "Avg. Efficiency", value: "78.4", unit: "% Peak", icon: <Zap size={18} className="text-yellow-400"/> },
    { label: "Stability", value: "99.9", unit: "Pass Rate", icon: <ShieldAlert size={18} className="text-emerald-400"/> },
    { label: "Analysed Ops", value: "24", unit: "Operators", icon: <Activity size={18} className="text-purple-400"/> }
  ];

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased font-sans overflow-x-hidden">
      {/* 1. 사이드바: 상태 제어 프롭스 전달 */}
      <ComputeSidebar 
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeOpId={opId}
      />

      <main className="flex-1 flex flex-col min-w-0 overflow-y-auto">
        {/* 2. 모바일 전용 헤더: md(768px) 이상에서 숨김 */}
        <header className="md:hidden sticky top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-emerald-400 tracking-tighter uppercase flex items-center gap-2">
              <Microscope size={18} /> Analysis Lab
            </div>
            <button 
              onClick={() => setIsSidebarOpen(true)} 
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200 active:scale-95 transition-transform"
            >
              <Menu size={20} />
            </button>
          </div>
        </header>

        {/* 3. 본문 컨텐츠 영역 */}
        <div className="p-6 sm:p-10 space-y-12 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          
          {/* 조건부 렌더링 영역 */}
          {kernelId ? (
            <KernelDetailView kernelData={kernelData} />
          ) : opId ? (
            <OpsComparisonView opData={opData} />
          ) : (
            /* 초기 Dashboard View */
            <>
              <section className="bg-[#1e293b] border border-slate-800 rounded-[3rem] p-10 sm:p-14 shadow-2xl relative overflow-hidden mt-4 md:mt-0">
                <div className="absolute -top-10 -right-10 text-[140px] font-black text-emerald-500/5 pointer-events-none uppercase">Analysis</div>
                <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-black uppercase mb-6">
                  <Microscope size={16} /> Kernel Understanding Pipeline
                </div>
                <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                  커널을 코드가 아닌 <br/><span className="text-emerald-400">실험의 대상</span>으로
                </h1>
                <p className="mt-8 text-slate-400 text-lg leading-relaxed max-w-3xl">
                  AICF Kernel Analysis System은 정적 코드 분석과 동적 프로파일링 데이터를 결합하여 하드웨어 병목 패턴을 식별합니다.
                </p>
              </section>

              <section className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
                {globalMetrics.map((m, i) => (
                  <div key={i} className="bg-[#1e293b] border border-slate-800 p-6 rounded-[2rem] shadow-xl flex flex-col justify-between h-40 group hover:border-emerald-500/30 transition-colors">
                    <div className="flex justify-between items-start">
                      <div className="p-3 bg-slate-800/50 rounded-2xl border border-slate-700 group-hover:bg-slate-700 transition-colors">{m.icon}</div>
                      <div className="text-slate-500 font-mono text-[10px] uppercase font-black tracking-widest">{m.label}</div>
                    </div>
                    <div>
                      <div className="text-3xl font-black text-white leading-none mb-1">{m.value}</div>
                      <div className="text-slate-500 text-[10px] font-mono uppercase">{m.unit}</div>
                    </div>
                  </div>
                ))}
              </section>

              {/* 정적/동적 분석 아키텍처 설명 섹션 */}
              <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 space-y-6 hover:bg-[#0f172a] transition-colors">
                  <div className="flex items-center gap-3"><FileCode className="text-blue-500" size={24} /><h3 className="text-xl font-black text-white uppercase">Static Analyzer</h3></div>
                  <p className="text-slate-500 text-sm leading-relaxed">CUDA 소스 구조 분석을 통해 Memory Stride와 Thread Mapping을 추론합니다.</p>
                </div>
                <div className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 space-y-6 hover:bg-[#0f172a] transition-colors">
                  <div className="flex items-center gap-3"><PieChart className="text-emerald-500" size={24} /><h3 className="text-xl font-black text-white uppercase">Dynamic Profiler</h3></div>
                  <p className="text-slate-500 text-sm leading-relaxed">NCU 하드웨어 메트릭을 수집하여 실제 런타임 성능과 병목을 판별합니다.</p>
                </div>
              </section>
            </>
          )}
        </div>
      </main>
    </div>
  );
}