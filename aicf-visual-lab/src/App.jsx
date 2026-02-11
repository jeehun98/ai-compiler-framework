// src/App.jsx
import React, { useState } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell
} from 'recharts';
import { Cpu, ChevronRight, Zap, Share2 } from 'lucide-react';
import { allOpsData } from './data/index.js'; // ← 경로 고정

export default function App() {
  const [selectedOpId, setSelectedOpId] = useState('GEMM');
  const data = allOpsData[selectedOpId];

  // -------------------------
  // HARD GUARD (절대 크래시 방지)
  // -------------------------
  if (!data) {
    return (
      <div className="p-10 text-red-300 bg-[#0f172a] h-screen">
        <h2 className="text-xl font-bold mb-4">Missing Op Data</h2>
        <div>selectedOpId: {selectedOpId}</div>
        <pre className="mt-4 text-xs text-slate-300">
          {JSON.stringify(allOpsData, null, 2)}
        </pre>
      </div>
    );
  }

  const latency = data.performance?.latency;
  if (!latency) {
    return (
      <div className="p-10 text-red-300 bg-[#0f172a] h-screen">
        <h2 className="text-xl font-bold mb-4">Missing performance.latency</h2>
        <div>op: {data.id}</div>
        <pre className="mt-4 text-xs text-slate-300">
          {JSON.stringify(data, null, 2)}
        </pre>
      </div>
    );
  }

  // -------------------------
  // Chart Data
  // -------------------------
  const latencyData = [
    { name: 'PyTorch', value: latency.pytorch ?? 0, color: '#64748b' },
    { name: 'Compile', value: latency.torch_compile ?? 0, color: '#94a3b8' },
    { name: 'AICF', value: latency.ours ?? 0, color: '#3b82f6' },
  ];

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-[#1e293b] border-r border-slate-700 p-6 flex flex-col">
        <h1 className="text-xl font-bold text-blue-400 mb-10 tracking-tight flex items-center gap-2">
          <Cpu size={24} /> AICF LAB
        </h1>
        <div className="space-y-2 flex-1 overflow-y-auto">
          {Object.keys(allOpsData).map(id => (
            <button
              key={id}
              onClick={() => setSelectedOpId(id)}
              className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all ${
                selectedOpId === id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'hover:bg-slate-800 text-slate-400'
              }`}
            >
              <span className="font-semibold uppercase tracking-wide">{id}</span>
              <ChevronRight size={14} />
            </button>
          ))}
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 p-10 overflow-y-auto space-y-12">
        <header>
          <span className="text-blue-400 font-mono text-sm uppercase tracking-widest">
            {data.category}
          </span>
          <h2 className="text-4xl font-extrabold mt-1">
            {data.id} Analysis
          </h2>
        </header>

        {/* Semantic */}
        <section className="space-y-4">
          <div className="flex items-center gap-2 text-blue-400 border-b border-slate-700 pb-2">
            <Share2 size={20} />
            <h3 className="text-xl font-bold uppercase italic">
              1. Semantic Analysis
            </h3>
          </div>

          <div className="grid grid-cols-2 gap-6">
            <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-700">
              <h4 className="text-slate-500 text-xs font-bold mb-4 uppercase">
                Mathematical Proposition
              </h4>
              <div className="bg-[#0f172a] p-8 rounded-xl flex items-center justify-center border border-slate-800">
                <code className="text-2xl text-blue-300 font-mono italic">
                  {data.semantic.formula}
                </code>
              </div>
            </div>

            <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-700">
              <h4 className="text-slate-500 text-xs font-bold mb-4 uppercase">
                Compiler IR Decomposition
              </h4>
              <div className="space-y-3">
                {data.semantic.decomposition.map((step, i) => (
                  <div
                    key={i}
                    className="flex items-center gap-4 bg-[#0f172a] p-3 px-5 rounded-lg border border-slate-800"
                  >
                    <span className="text-blue-500 font-black font-mono">
                      STEP_0{i + 1}
                    </span>
                    <span className="text-slate-300">{step}</span>
                  </div>
                ))}
              </div>
            </div>
          </div>
        </section>

        {/* Kernel */}
        <section className="space-y-4 pb-10">
          <div className="flex items-center gap-2 text-emerald-400 border-b border-slate-700 pb-2">
            <Zap size={20} />
            <h3 className="text-xl font-bold uppercase italic">
              2. Kernel Optimization
            </h3>
          </div>

          <div className="grid grid-cols-3 gap-6 mb-6">
            <MetricCard title="Peak Throughput" value={data.optimization.throughput} color="text-emerald-400" />
            <MetricCard title="GPU Occupancy" value={`${data.optimization.occupancy}%`} color="text-blue-400" />
            <MetricCard title="Memory Reuse" value={data.optimization.memory_reuse} color="text-purple-400" />
          </div>

          <div className="grid grid-cols-2 gap-8">
            <div className="bg-[#1e293b] p-8 rounded-2xl border border-slate-700">
              <h4 className="text-slate-500 text-xs font-bold mb-8 uppercase">
                Performance Benchmark (ms)
              </h4>
              <div className="h-60">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={latencyData} layout="vertical" margin={{ left: 20 }}>
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" stroke="#94a3b8" width={80} />
                    <Tooltip />
                    <Bar dataKey="value" barSize={24} radius={[0, 4, 4, 0]}>
                      {latencyData.map((e, i) => (
                        <Cell key={i} fill={e.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>
            </div>

            <div className="bg-[#1e293b] p-8 rounded-2xl border border-slate-700 flex flex-col">
              <div className="flex justify-between items-center mb-6">
                <h4 className="text-slate-500 text-xs font-bold uppercase">
                  Kernel Implementation
                </h4>
                <span className="text-[10px] bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 px-2 py-1 rounded-md font-bold">
                  {data.optimization.strategy}
                </span>
              </div>
              <pre className="flex-1 bg-[#0f172a] p-5 rounded-xl text-[13px] font-mono text-emerald-300 overflow-x-auto border border-slate-800">
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
function MetricCard({ title, value, color }) {
  return (
    <div className="bg-[#1e293b] p-6 rounded-2xl border border-slate-700">
      <p className="text-slate-500 text-[10px] mb-2 uppercase font-black">
        {title}
      </p>
      <p className={`text-3xl font-black font-mono ${color}`}>
        {value}
      </p>
    </div>
  );
}
