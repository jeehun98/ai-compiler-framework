import React from 'react';
import { X, TrendingUp, Cpu, Activity, Zap, CheckCircle2 } from 'lucide-react';

const KernelDeepDive = ({ isOpen, onClose, data }) => {
  if (!isOpen || !data) return null;

  // 병합된 데이터에서 필요한 정보 추출
  const history = data.kernel_evolution || [];
  const profiling = data.profiling_report || {};
  const analysis = data.analysis || "";

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-sm animate-in fade-in duration-200">
      <div className="bg-[#1e293b] w-full max-w-4xl max-h-[90vh] rounded-[2rem] border border-slate-700 shadow-2xl overflow-hidden flex flex-col">
        
        {/* Header */}
        <div className="flex items-center justify-between p-8 border-b border-slate-700 bg-[#0f172a]">
          <div>
            <div className="flex items-center gap-3 text-emerald-400 mb-2">
              <Activity size={24} />
              <h2 className="text-sm font-black uppercase tracking-[0.3em]">Optimization Chronicle</h2>
            </div>
            <h1 className="text-3xl font-black text-white italic tracking-tighter">
              {data.id} Kernel Evolution
            </h1>
          </div>
          <button 
            onClick={onClose}
            className="p-3 bg-slate-800 rounded-full hover:bg-slate-700 text-slate-400 hover:text-white transition-all"
          >
            <X size={24} />
          </button>
        </div>

        {/* Content Scroll Area */}
        <div className="flex-1 overflow-y-auto p-8 space-y-10 scrollbar-thin scrollbar-thumb-slate-600 scrollbar-track-transparent">
          
          {/* 1. Evolution Timeline */}
          <section>
            <h3 className="text-slate-400 font-bold uppercase tracking-widest text-xs mb-6 flex items-center gap-2">
              <TrendingUp size={16} /> Version History
            </h3>
            <div className="space-y-6 relative pl-4 border-l-2 border-slate-700 ml-2">
              {history.map((ver, idx) => (
                <div key={idx} className="relative pl-8 group">
                  {/* Timeline Dot */}
                  <div className={`absolute -left-[21px] top-1 w-4 h-4 rounded-full border-2 ${idx === history.length - 1 ? 'bg-emerald-500 border-emerald-500 shadow-[0_0_10px_rgba(16,185,129,0.5)]' : 'bg-[#1e293b] border-slate-600'}`} />
                  
                  <div className="bg-[#0f172a] p-6 rounded-2xl border border-slate-800 group-hover:border-emerald-500/30 transition-all">
                    <div className="flex justify-between items-start mb-3">
                      <div>
                        <span className={`text-[10px] font-black px-2 py-1 rounded-md mr-3 ${idx === history.length - 1 ? 'bg-emerald-500/10 text-emerald-400' : 'bg-slate-800 text-slate-500'}`}>
                          {ver.version}
                        </span>
                        <span className="text-lg font-bold text-slate-200">{ver.tag}</span>
                      </div>
                      <div className="text-right">
                        <span className="text-2xl font-black text-emerald-400 font-mono block">{ver.throughput}</span>
                      </div>
                    </div>
                    <p className="text-sm text-slate-400 leading-relaxed">
                      {ver.description}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* 2. Profiling Report Grid */}
          <section>
             <h3 className="text-slate-400 font-bold uppercase tracking-widest text-xs mb-6 flex items-center gap-2">
              <Cpu size={16} /> Profiling Metrics (Latest)
            </h3>
            <div className="grid grid-cols-2 md:grid-cols-5 gap-4">
              {Object.entries(profiling).map(([key, value]) => (
                <div key={key} className="bg-[#0f172a]/50 p-4 rounded-xl border border-slate-800 text-center">
                  <p className="text-[9px] text-slate-500 font-black uppercase tracking-tight mb-2 truncate" title={key}>
                    {key.replace(/_/g, ' ')}
                  </p>
                  <p className="text-xl font-black text-white font-mono">{value}</p>
                </div>
              ))}
            </div>
          </section>

          {/* 3. Analysis Text */}
          <section className="bg-emerald-900/10 p-6 rounded-2xl border border-emerald-500/20">
            <h3 className="text-emerald-400 font-bold uppercase tracking-widest text-xs mb-3 flex items-center gap-2">
              <CheckCircle2 size={16} /> Analysis & Conclusion
            </h3>
            <p className="text-emerald-100/80 text-sm leading-relaxed font-medium">
              {analysis}
            </p>
          </section>
        </div>
      </div>
    </div>
  );
};

export default KernelDeepDive;