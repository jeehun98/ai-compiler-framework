import React from 'react';
import { X, History, TrendingUp, BarChart3, Target } from 'lucide-react';

export default function KernelDeepDive({ isOpen, onClose, data }) {
  if (!isOpen || !data) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 md:p-12 bg-[#0b1120]/95 backdrop-blur-md animate-in fade-in duration-300">
      <div className="relative w-full max-w-6xl max-h-full bg-[#1e293b] rounded-[3rem] border border-slate-700 shadow-2xl overflow-hidden flex flex-col">
        
        {/* Modal Header */}
        <div className="p-10 border-b border-slate-800 flex justify-between items-center bg-[#1e293b]">
          <div>
            <h3 className="text-4xl font-black italic uppercase text-emerald-400 tracking-tighter">
              {data.id} Optimization Chronicle
            </h3>
            <p className="text-slate-500 text-xs mt-2 uppercase tracking-[0.3em] font-black italic">
              Hardware Performance & Engineering evolution
            </p>
          </div>
          <button onClick={onClose} className="p-4 hover:bg-slate-800 rounded-full text-slate-400 transition-all">
            <X size={32} />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-12 overflow-y-auto space-y-16 scrollbar-hide">
          {/* 최적화 히스토리 섹션 */}
          <section>
            <div className="flex items-center gap-3 mb-12 text-slate-300">
              <History size={24} className="text-emerald-500" />
              <h4 className="text-xl font-black uppercase tracking-tight italic">Optimization Milestone</h4>
            </div>
            <div className="relative space-y-8 pl-10 border-l-2 border-slate-800 ml-4">
              {data.kernel_evolution?.map((evo, i) => (
                <div key={i} className="relative p-8 bg-[#0f172a] rounded-[2rem] border border-slate-800 group transition-all shadow-inner">
                  <div className="absolute -left-[51px] top-1/2 -translate-y-1/2 w-5 h-5 bg-emerald-500 rounded-full border-4 border-[#1e293b]" />
                  <div className="flex flex-col md:flex-row justify-between mb-4">
                    <h5 className="text-2xl font-black text-white italic">{evo.tag}</h5>
                    <span className="text-2xl font-black text-emerald-400 font-mono">{evo.throughput}</span>
                  </div>
                  <p className="text-sm text-slate-400 leading-relaxed">{evo.description}</p>
                </div>
              ))}
            </div>
          </section>

          {/* 하드웨어 리소스 분석 섹션 */}
          <section className="grid grid-cols-1 lg:grid-cols-2 gap-10 pb-10">
            <div className="bg-[#0f172a] p-10 rounded-[2.5rem] border border-slate-800">
              <h4 className="text-xs font-black text-slate-500 uppercase mb-10 flex items-center gap-2 tracking-widest">
                <BarChart3 size={20} /> Compute Unit Utilization
              </h4>
              <div className="space-y-8">
                {Object.entries(data.profiling_report ?? {}).map(([key, val]) => (
                  <div key={key} className="space-y-2">
                    <div className="flex justify-between text-[11px] uppercase font-black tracking-widest">
                      <span className="text-slate-500">{key.replace(/_/g, ' ')}</span>
                      <span className="text-emerald-400">{val}</span>
                    </div>
                    <div className="h-2 bg-slate-900 rounded-full overflow-hidden border border-slate-800">
                      <div className="h-full bg-emerald-500" style={{ width: val }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
            
            <div className="bg-[#0f172a] p-10 rounded-[2.5rem] border border-slate-800 flex flex-col items-center justify-center">
              
              <p className="text-sm text-slate-400 mt-8 max-w-xs text-center italic">
                "AICF kernel achieves near-peak performance by maximizing TensorCore utilization."
              </p>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}