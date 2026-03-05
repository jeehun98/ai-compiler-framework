import React from 'react';
import { Microscope, Activity, Database, ShieldCheck, Zap, Cpu, Code2, Info, Layers } from 'lucide-react';

export default function KernelDetailView({ kernelData }) {
  const m = kernelData.metrics.metrics;
  const algo = kernelData.algorithm;
  const bp = kernelData.blueprint;

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
          <span className="px-4 py-1 bg-emerald-500/10 text-emerald-400 rounded-full text-xs font-bold border border-emerald-500/20">
            {kernelData.tag}
          </span>
        </div>
      </header>

      {/* 1. Performance Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <MetricCard label="SM Throughput" value={m["sm__throughput.avg.pct_of_peak_sustained_elapsed"].val} unit="%" icon={<Activity size={18}/>} />
        <MetricCard label="Active Warps" value={m["sm__warps_active.avg.pct_of_peak_sustained_active"].val} unit="%" icon={<Zap size={18}/>} />
        <MetricCard label="Compute Time" value={m["gpu__time_duration.sum"].val} unit="us" icon={<Database size={18}/>} />
        <MetricCard label="Occupancy" value={78.4} unit="%" icon={<ShieldCheck size={18}/>} />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        <div className="lg:col-span-2 space-y-6">
          {/* 2. Execution Blueprint (커널 알고리즘 상세) */}
          <div className="bg-[#1e293b] border border-slate-800 rounded-[3rem] p-10 overflow-hidden relative shadow-2xl">
            <div className="absolute top-8 left-10 text-slate-500 font-mono text-[10px] uppercase flex items-center gap-2">
              <Code2 size={12} /> Implementation Blueprint
            </div>
            
            <div className="mt-10 grid grid-cols-1 md:grid-cols-2 gap-12">
              <div className="space-y-6">
                <div>
                  <h3 className="text-2xl font-black text-white mb-3">{algo?.title}</h3>
                  <p className="text-slate-400 text-sm leading-relaxed">{algo?.logic}</p>
                </div>

                {/* Core Instruction 스니펫 */}
                <div className="bg-[#0f172a] rounded-2xl p-5 border border-slate-700/50 relative group">
                  <div className="text-[9px] text-emerald-500 font-black uppercase mb-3 tracking-tighter">Core Device Instruction</div>
                  <code className="text-emerald-400 font-mono text-xs leading-relaxed block">
                    {bp?.code_snippet}
                  </code>
                  <div className="absolute top-4 right-4 text-slate-700 group-hover:text-emerald-500/30 transition-colors">
                    <Layers size={24} />
                  </div>
                </div>
              </div>

              {/* 하드웨어 스펙 테이블 */}
              <div className="bg-slate-900/40 rounded-[2.5rem] border border-slate-800/50 p-8 flex flex-col justify-center space-y-5 shadow-inner">
                <BlueprintRow label="Mem Access" value={bp?.mem_access} />
                <BlueprintRow label="Instruction" value={bp?.instruction} />
                <BlueprintRow label="Vector Width" value={`${bp?.vector_width} element(s)`} color="text-emerald-400" />
                <BlueprintRow label="Alignment" value={bp?.alignment_req} />
                <div className="pt-4 border-t border-slate-800 mt-2 text-center text-[10px] text-slate-500 font-mono">
                  Target Arch: NVIDIA Ampere+
                </div>
              </div>
            </div>
          </div>

          {/* Roofline Model Section */}
          <div className="bg-[#1e293b] border border-slate-800 rounded-[3rem] p-10 h-80 relative flex flex-col">
            <div className="absolute top-8 left-10 text-slate-500 font-mono text-[10px] uppercase flex items-center gap-2">
              <Activity size={12} /> Roofline Analysis (LOG-LOG)
            </div>
            
            <div className="mt-8 flex-1 w-full border-l-2 border-b-2 border-slate-700 relative overflow-hidden">
              {/* 1. Peak Performance Lines (The "Roof") */}
              <svg className="absolute inset-0 w-full h-full">
                {/* Memory Wall (경사면) */}
                <line x1="0" y1="100%" x2="70%" y2="20%" stroke="#334155" strokeWidth="2" strokeDasharray="4" />
                {/* Compute Peak (평면) */}
                <line x1="70%" y1="20%" x2="100%" y2="20%" stroke="#334155" strokeWidth="2" strokeDasharray="4" />
                
                {/* 좌표축 라벨 */}
                <text x="5" y="15" fill="#475569" fontSize="10" fontFamily="monospace">GFLOPS</text>
                <text x="85%" y="95%" fill="#475569" fontSize="10" fontFamily="monospace">Intensity</text>
              </svg>

              {/* 2. Kernel Position Dot */}
              {/* Intensity가 낮으면 왼쪽(Memory-bound), 높으면 오른쪽(Compute-bound) */}
              <div 
                className="absolute transition-all duration-1000 ease-out"
                style={{
                  bottom: kernelData.id === 'f16x2' ? '45%' : '20%', 
                  left: kernelData.id === 'f32' ? '15%' : '40%',
                }}
              >
                {/* Glow Effect */}
                <div className="absolute inset-0 bg-emerald-500 blur-xl opacity-50 animate-pulse" />
                {/* Target Point */}
                <div className="relative w-4 h-4 bg-emerald-500 rounded-full border-2 border-white shadow-lg" />
                
                {/* Tooltip on Chart */}
                <div className="absolute left-6 -top-2 bg-slate-900 border border-slate-700 px-2 py-1 rounded text-[9px] text-white whitespace-nowrap shadow-xl">
                  Current Op: {kernelData.id}
                </div>
              </div>
            </div>

            {/* 3. Status Footer */}
            <div className="mt-4 flex justify-between items-center">
              <p className="text-slate-500 text-[10px] italic">
                {kernelData.id === 'f32' 
                  ? "High memory pressure due to 4-byte access" 
                  : "Optimization detected: Improved arithmetic intensity"}
              </p>
              <div className="flex gap-4">
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-slate-700 rounded-full" />
                  <span className="text-[9px] text-slate-500 uppercase">Peak</span>
                </div>
                <div className="flex items-center gap-1">
                  <div className="w-2 h-2 bg-emerald-500 rounded-full" />
                  <span className="text-[9px] text-slate-500 uppercase">Actual</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* 3. Expert Insights Sidebar */}
        <div className="bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8 h-fit sticky top-8">
          <div className="flex items-center gap-2 text-blue-400 font-mono text-[10px] font-black uppercase tracking-widest mb-6">
            <Info size={14} /> Optimization Insights
          </div>
          <div className="space-y-4">
            {kernelData.insights?.map((info, i) => (
              <div key={i} className="p-5 bg-slate-900/80 rounded-3xl border border-slate-800 flex gap-4 hover:border-slate-600 transition-colors group">
                <div className="w-2 h-2 rounded-full bg-emerald-500 mt-1.5 shrink-0 shadow-[0_0_10px_#10b981] group-hover:scale-125 transition-transform" />
                <p className="text-[12px] text-slate-400 leading-relaxed font-medium">{info}</p>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// Sub-components
function MetricCard({ label, value, unit, icon }) {
  return (
    <div className="bg-[#1e293b] border border-slate-800 p-7 rounded-[2.5rem] group hover:border-emerald-500/50 transition-all shadow-xl">
      <div className="flex justify-between items-start mb-5">
        <div className="p-2.5 bg-slate-800 rounded-2xl text-slate-400 group-hover:text-emerald-400 group-hover:bg-emerald-500/10 transition-all">{icon}</div>
        <div className="text-slate-500 font-mono text-[10px] uppercase font-black tracking-tighter">{label}</div>
      </div>
      <div className="flex items-baseline gap-1">
        <span className="text-4xl font-black text-white tracking-tight">{typeof value === 'number' ? value.toFixed(1) : value}</span>
        <span className="text-slate-500 text-xs font-mono font-bold">{unit}</span>
      </div>
    </div>
  );
}

function BlueprintRow({ label, value, color = "text-slate-300" }) {
  return (
    <div className="flex justify-between items-center py-1 border-b border-slate-800/50 last:border-0">
      <span className="text-[10px] font-mono text-slate-500 uppercase font-black tracking-tight">{label}</span>
      <span className={`text-xs font-bold ${color}`}>{value}</span>
    </div>
  );
}