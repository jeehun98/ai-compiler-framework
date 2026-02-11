// src/App.jsx
import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Cpu, Activity, ChevronRight, Zap, Share2 } from 'lucide-react';
import { allOpsData } from './data'; // 분리된 데이터 가져오기

export default function App() {
  const [selectedOpId, setSelectedOpId] = useState('GEMM');
  const data = allOpsData[selectedOpId];

  // 차트 데이터 변환 로직
  const latencyData = [
    { name: 'PyTorch', value: data.performance.latency.pytorch, color: '#64748b' },
    { name: 'Compile', value: data.performance.latency.torch_compile, color: '#94a3b8' },
    { name: 'AICF', value: data.performance.latency.ours, color: '#3b82f6' },
  ];

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-[#1e293b] border-r border-slate-700 p-6 flex flex-col">
        <h1 className="text-xl font-bold text-blue-400 mb-10 tracking-tight flex items-center gap-2">
          <Cpu className="text-blue-500" size={24} /> AICF LAB
        </h1>
        <div className="space-y-2 flex-1 overflow-y-auto">
          {Object.keys(allOpsData).map(id => (
            <button
              key={id}
              onClick={() => setSelectedOpId(id)}
              className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all ${
                selectedOpId === id ? 'bg-blue-600 text-white shadow-lg' : 'hover:bg-slate-800 text-slate-400'
              }`}
            >
              <span className="font-semibold uppercase tracking-wide">{id}</span>
              <ChevronRight size={14} />
            </button>
          ))}
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-10 overflow-y-auto space-y-12">
        <header>
          <span className="text-blue-400 font-mono text-sm uppercase tracking-widest">{data.category}</span>
          <h2 className="text-4xl font-extrabold mt-1">{data.id} Analysis</h2>
        </header>

        {/* --- SECTION 1: Semantic Analysis --- */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-blue-400 border-b border-slate-700 pb-2">
            <Share2 size={20} />
            <h3 className="text-xl font-bold uppercase tracking-tight italic">1. Semantic Analysis</h3>
          </div>
          
          <div className="grid grid-cols-2 gap-6">
            <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-700">
              <h4 className="text-slate-500 text-xs font-bold mb-4 uppercase tracking-widest">Mathematical Proposition</h4>
              <div className="bg-[#0f172a] p-8 rounded-xl flex items-center justify-center border border-slate-800 shadow-inner">
                <code className="text-2xl text-blue-300 font-mono italic">{data.semantic.formula}</code>
              </div>
            </div>
            
            <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-700">
              <h4 className="text-slate-500 text-xs font-bold mb-4 uppercase tracking-widest">Compiler IR Decomposition</h4>
              <div className="space-y-3">
                {data.semantic.decomposition.map((step, i) => (
                  <div key={i} className="flex items-center gap-4 bg-[#0f172a] p-3 px-5 rounded-lg border border-slate-800">
                    <span className="text-blue-500 font-black font-mono">STEP_0{i+1}</span>
                    <span className="text-slate-300 font-medium">{step}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* --- SECTION 2: Kernel Optimization --- */}
        <section className="space-y-4 pb-10">
          <div className="flex items-center gap-2 text-emerald-400 border-b border-slate-700 pb-2">
            <Zap size={20} />
            <h3 className="text-xl font-bold uppercase tracking-tight italic">2. Kernel Optimization</h3>
          </div>

          <div className="grid grid-cols-3 gap-6 mb-6">
            <MetricCard title="Peak Throughput" value={data.optimization.throughput} color="text-emerald-400" />
            <MetricCard title="GPU Occupancy" value={`${data.optimization.occupancy}%`} color="text-blue-400" />
            <MetricCard title="Memory Reuse" value={data.optimization.memory_reuse} color="text-purple-400" />
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="bg-[#1e293b] p-8 rounded-2xl border border-slate-700">
              <h4 className="text-slate-500 text-xs font-bold mb-8 uppercase tracking-widest">Performance Benchmark (ms)</h4>
              <div className="h-60">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={latencyData} layout="vertical" margin={{ left: 20 }}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={12} width={80} />
                    <Tooltip cursor={{fill: '#2d3748'}} contentStyle={{backgroundColor: '#1e293b', border: 'none', borderRadius: '10px'}} />
                    <Bar dataKey="value" radius={[0, 4, 4, 0]} barSize={24}>
                      {latencyData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-[#1e293b] p-8 rounded-2xl border border-slate-700 flex flex-col">
              <div className="flex justify-between items-center mb-6">
                <h4 className="text-slate-500 text-xs font-bold uppercase tracking-widest">Kernel Implementation</h4>
                <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-1 rounded-md font-bold">
                  {data.optimization.strategy}
                </span>
              </div>
              <pre className="flex-1 bg-[#0f172a] p-5 rounded-xl text-[13px] font-mono text-emerald-300 overflow-x-auto border border-slate-800 leading-relaxed scrollbar-thin scrollbar-thumb-slate-700">
                <code>{data.cudaCode}</code>
              </pre>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

// 재사용 가능한 메트릭 카드 컴포넌트
function MetricCard({ title, value, color }) {
  return (
    <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-700 hover:border-slate-500 transition-colors shadow-lg">
      <p className="text-slate-500 text-[10px] mb-2 uppercase font-black tracking-widest">{title}</p>
      <p className={`text-3xl font-black font-mono ${color}`}>{value}</p>
    </div>
  );
}