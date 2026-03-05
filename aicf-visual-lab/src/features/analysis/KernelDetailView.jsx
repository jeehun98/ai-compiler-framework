import React from 'react';
import { Microscope, Activity, Database, ShieldCheck, Zap } from 'lucide-react';

export default function KernelDetailView({ kernelData }) {
  const m = kernelData.metrics.metrics;

  return (
    <div className="space-y-8 animate-in slide-in-from-bottom-6 duration-700">
      <header className="flex justify-between items-end">
        <div>
          <div className="flex items-center gap-2 text-emerald-400 font-mono text-[10px] font-black uppercase tracking-widest mb-2">
            <Microscope size={14} /> Laboratory Analysis Report
          </div>
          <h2 className="text-4xl font-black text-white">{kernelData.name}</h2>
        </div>
        <div className="flex gap-2">
          <span className="px-4 py-1 bg-slate-800 text-slate-400 rounded-full text-xs font-bold border border-slate-700">
            {kernelData.tag}
          </span>
        </div>
      </header>

      {/* Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard label="SM Throughput" value={m["sm__throughput.avg.pct_of_peak_sustained_elapsed"].val} unit="%" icon={<Activity size={18}/>} />
        <MetricCard label="Active Warps" value={m["sm__warps_active.avg.pct_of_peak_sustained_active"].val} unit="%" icon={<Zap size={18}/>} />
        <MetricCard label="Compute Time" value={m["gpu__time_duration.sum"].val} unit="us" icon={<Database size={18}/>} />
        <MetricCard label="Occupancy" value={78.4} unit="%" icon={<ShieldCheck size={18}/>} />
      </div>

      {/* Roofline Model Section */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 bg-[#1e293b] border border-slate-800 rounded-[3rem] p-10 h-96 relative flex flex-col items-center justify-center">
          <div className="absolute top-8 left-10 text-slate-500 font-mono text-[10px] uppercase">Roofline Analysis</div>
          <div className="w-full h-full border-l border-b border-slate-700 relative">
             {/* Roofline Visualization */}
             <div className="absolute bottom-[18%] left-[82%] w-6 h-6 bg-emerald-500 rounded-full shadow-[0_0_40px_#10b981] animate-pulse" />
          </div>
          <p className="mt-4 text-slate-500 text-xs italic">Memory-bound bottleneck identified</p>
        </div>

        <div className="bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8">
          <h4 className="text-white font-black uppercase text-sm mb-6">Expert Insights</h4>
          <div className="space-y-4">
            {kernelData.insights?.map((info, i) => (
              <div key={i} className="p-4 bg-slate-900/50 rounded-2xl border border-slate-800 flex gap-3">
                <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5 shrink-0" />
                <p className="text-[11px] text-slate-400 leading-relaxed">{info}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

function MetricCard({ label, value, unit, icon }) {
  return (
    <div className="bg-[#1e293b] border border-slate-800 p-6 rounded-[2rem] group hover:border-emerald-500/50 transition-all">
      <div className="flex justify-between items-start mb-4">
        <div className="p-2 bg-slate-800 rounded-xl text-slate-400 group-hover:text-emerald-400 transition-colors">{icon}</div>
        <div className="text-slate-500 font-mono text-[9px] uppercase font-black">{label}</div>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-3xl font-black text-white">{value.toFixed(1)}</span>
        <span className="text-slate-500 text-xs font-mono">{unit}</span>
      </div>
    </div>
  );
}