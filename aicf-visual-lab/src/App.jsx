import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { Cpu, ChevronRight, Zap, Share2, Box, Layers, Target, ShieldCheck, Activity } from 'lucide-react';

import { allOpsData } from './data/index.js';

export default function App() {
  const [selectedOpId, setSelectedOpId] = useState('GEMM');
  const data = allOpsData[selectedOpId];

  // --- HARD GUARD ---
  if (!data) return <div className="p-10 text-red-300 bg-[#0f172a] h-screen">Loading Data...</div>;

  const latency = data.performance?.latency;
  const latencyData = [
    { name: 'PyTorch', value: latency?.pytorch ?? 0, color: '#64748b' },
    { name: 'Compile', value: latency?.torch_compile ?? 0, color: '#94a3b8' },
    { name: 'AICF', value: latency?.ours ?? 0, color: '#3b82f6' },
  ];

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-72 bg-[#1e293b] border-r border-slate-700 p-6 flex flex-col">
        <h1 className="text-xl font-bold text-blue-400 mb-10 tracking-tight flex items-center gap-2">
          <Cpu size={24} className="text-blue-500" /> AICF VISUAL LAB
        </h1>
        <div className="space-y-2 flex-1 overflow-y-auto">
          {Object.keys(allOpsData).map(id => (
            <button
              key={id}
              onClick={() => setSelectedOpId(id)}
              className={`w-full flex flex-col items-start px-4 py-3 rounded-xl transition-all ${
                selectedOpId === id ? 'bg-blue-600 text-white shadow-lg' : 'hover:bg-slate-800 text-slate-400'
              }`}
            >
              <div className="flex justify-between w-full items-center">
                <span className="font-bold uppercase tracking-wider text-sm">{id}</span>
                <ChevronRight size={14} opacity={selectedOpId === id ? 1 : 0.3} />
              </div>
              <span className={`text-[10px] mt-1 ${selectedOpId === id ? 'text-blue-100' : 'text-slate-500'}`}>
                {allOpsData[id].category}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-10 overflow-y-auto space-y-12 bg-gradient-to-b from-[#0f172a] to-[#1e293b]/20">
        <header className="flex justify-between items-end border-b border-slate-800 pb-8">
          <div>
            <span className="text-blue-400 font-mono text-xs uppercase tracking-[0.3em] font-bold">
              {data.category}
            </span>
            <h2 className="text-5xl font-black mt-2 tracking-tighter italic">
              {data.id} <span className="text-slate-500 font-light not-italic text-3xl ml-2">Spec.v1</span>
            </h2>
          </div>
          <div className="flex gap-3">
             <div className="px-4 py-2 bg-blue-500/10 border border-blue-500/30 rounded-full text-blue-400 text-xs font-bold uppercase tracking-widest">Semantic Layer</div>
             <div className="px-4 py-2 bg-emerald-500/10 border border-emerald-500/30 rounded-full text-emerald-400 text-xs font-bold uppercase tracking-widest">Physical Layer</div>
          </div>
        </header>

        {/* SECTION 1: Semantic Analysis (상세 사양 반영) */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 text-blue-400">
            <Share2 size={24} />
            <h3 className="text-2xl font-black uppercase tracking-tight">1. Semantic Deep-Dive</h3>
          </div>

          <div className="grid grid-cols-12 gap-6">
            {/* Mathematical Proposition */}
            <div className="col-span-8 bg-[#1e293b] p-8 rounded-3xl border border-slate-700 shadow-xl">
              <div className="flex items-center gap-2 mb-6 text-slate-400">
                <Target size={16} />
                <h4 className="text-xs font-black uppercase tracking-widest">Canonical Form & Interpretation</h4>
              </div>
              <div className="flex flex-col items-center justify-center space-y-8 py-4">
                <div className="bg-[#0f172a] p-10 rounded-2xl border border-slate-800 w-full text-center shadow-inner">
                  <code className="text-4xl text-blue-300 font-mono italic tracking-widest">
                    {data.semantic.formula}
                  </code>
                </div>
                <div className="grid grid-cols-3 gap-8 w-full">
                  <div className="text-center p-4 rounded-xl bg-slate-800/50 border border-slate-700">
                    <p className="text-blue-400 font-bold text-lg">A (Samples)</p>
                    <p className="text-xs text-slate-500 mt-1 italic">Input Data Space</p>
                  </div>
                  <div className="text-center p-4 rounded-xl bg-slate-800/50 border border-slate-700">
                    <p className="text-purple-400 font-bold text-lg">B (Hypothesis)</p>
                    <p className="text-xs text-slate-500 mt-1 italic">K-Dimension Search</p>
                  </div>
                  <div className="text-center p-4 rounded-xl bg-slate-800/50 border border-slate-700">
                    <p className="text-emerald-400 font-bold text-lg">C (State)</p>
                    <p className="text-xs text-slate-500 mt-1 italic">Projection Result</p>
                  </div>
                </div>
              </div>
            </div>

            {/* Semantic Attributes */}
            <div className="col-span-4 space-y-6">
              <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-700">
                <h4 className="text-slate-500 text-xs font-black mb-6 uppercase tracking-widest flex items-center gap-2">
                  <ShieldCheck size={14} /> Semantic Constraints
                </h4>
                <div className="space-y-4">
                  {data.semantic.attributes?.map((attr, i) => (
                    <div key={i} className="flex justify-between items-center p-3 bg-[#0f172a] rounded-xl border border-slate-800">
                      <span className="text-xs text-slate-400 font-bold">{attr.label}</span>
                      <span className="text-xs text-blue-400 font-mono font-bold">{attr.value}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* Rewrite Rules (Rule 6.5, 6.6 등) */}
          <div className="bg-[#1e293b] p-8 rounded-3xl border border-slate-700">
             <h4 className="text-slate-500 text-xs font-black mb-6 uppercase tracking-widest">Semantic Rewrite Rules (Pre-Lowering)</h4>
             <div className="grid grid-cols-3 gap-4">
                {data.semantic.rules?.map((rule, i) => (
                  <div key={i} className="p-4 bg-blue-500/5 border border-blue-500/20 rounded-2xl flex gap-3 items-start">
                    <div className="bg-blue-500 text-[#0f172a] p-1 rounded font-black text-[10px]">R{i+1}</div>
                    <p className="text-xs text-slate-300 leading-relaxed font-medium">{rule}</p>
                  </div>
                ))}
             </div>
          </div>
        </section>

        {/* SECTION 2: Physical Optimization */}
        <section className="space-y-6 pb-20">
          <div className="flex items-center gap-3 text-emerald-400">
            <Zap size={24} />
            <h3 className="text-2xl font-black uppercase tracking-tight">2. Physical Optimization</h3>
          </div>

          <div className="grid grid-cols-4 gap-6">
            <MetricCard title="Peak Throughput" value={data.optimization.throughput} color="text-emerald-400" icon={<Activity size={14}/>} />
            <MetricCard title="K-Hypothesis Speed" value={`${data.optimization.occupancy}%`} color="text-blue-400" icon={<Box size={14}/>} />
            <MetricCard title="Memory Reuse" value={data.optimization.memory_reuse} color="text-purple-400" icon={<Layers size={14}/>} />
            <MetricCard title="Optimization" value={data.optimization.strategy} color="text-white" icon={<ShieldCheck size={14}/>} />
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="bg-[#1e293b] p-8 rounded-3xl border border-slate-700">
              <h4 className="text-slate-500 text-xs font-black mb-8 uppercase tracking-widest text-center">Physical Benchmark (ms)</h4>
              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={latencyData} layout="vertical">
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={12} width={80} />
                    <Tooltip 
                      cursor={{fill: '#2d3748'}} 
                      contentStyle={{backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px'}} 
                    />
                    <Bar dataKey="value" barSize={32} radius={[0, 8, 8, 0]}>
                      {latencyData.map((e, i) => (
                        <Cell key={i} fill={e.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-[#1e293b] p-8 rounded-3xl border border-slate-700 flex flex-col shadow-2xl">
              <h4 className="text-slate-500 text-xs font-black mb-6 uppercase tracking-widest flex justify-between items-center">
                Generated CUDA Kernel 
                <span className="text-[10px] text-emerald-400 font-mono tracking-normal lowercase">semantic_aware_v1.cu</span>
              </h4>
              <pre className="flex-1 bg-[#0f172a] p-6 rounded-2xl text-[13px] font-mono text-emerald-300 overflow-x-auto border border-slate-800 leading-relaxed shadow-inner scrollbar-thin scrollbar-thumb-slate-700">
                <code>{data.cudaCode}</code>
              </pre>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

// -------------------------
// Sub-Components
// -------------------------
function MetricCard({ title, value, color, icon }) {
  return (
    <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-700 hover:scale-[1.02] transition-transform duration-300">
      <div className="flex items-center gap-2 text-slate-500 mb-3">
        {icon}
        <p className="text-[10px] uppercase font-black tracking-[0.2em]">{title}</p>
      </div>
      <p className={`text-2xl font-black font-mono tracking-tighter ${color}`}>
        {value}
      </p>
    </div>
  );
}