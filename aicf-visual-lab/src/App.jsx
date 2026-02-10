import React, { useState } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import { Cpu, Activity, Code, ChevronRight, Database } from 'lucide-react';

// 내부 Mock Data (파일 분리 전 우선 통합)
const mockOpsData = {
  GEMM: {
    id: "GEMM",
    category: "Linear Layers",
    performance: {
      latency: { ours: 120, pytorch: 210, torch_compile: 155 },
      throughput: "84.2 TFLOPS",
      occupancy: "92%",
    },
    cudaCode: `__global__ void gemm_kernel(...) {\n  __shared__ float tile[32][32];\n  // Tiling logic\n}`
  },
  ReLU: {
    id: "ReLU",
    category: "Activations",
    performance: {
      latency: { ours: 15, pytorch: 25, torch_compile: 18 },
      throughput: "12.5 TFLOPS",
      occupancy: "98%",
    },
    cudaCode: `__global__ void relu_kernel(...) {\n  X[idx] = fmaxf(0.0f, X[idx]);\n}`
  }
};

export default function App() {
  const [selectedOpId, setSelectedOpId] = useState('GEMM');
  const data = mockOpsData[selectedOpId];

  const latencyData = [
    { name: 'PyTorch', value: data.performance.latency.pytorch, color: '#64748b' },
    { name: 'Compile', value: data.performance.latency.torch_compile, color: '#94a3b8' },
    { name: 'AICF', value: data.performance.latency.ours, color: '#3b82f6' },
  ];

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 font-sans overflow-hidden">
      {/* Sidebar */}
      <aside className="w-64 bg-[#1e293b] border-r border-slate-700 p-6">
        <h1 className="text-xl font-bold text-blue-400 mb-10 tracking-tight">AICF VISUAL LAB</h1>
        <div className="space-y-2">
          {Object.keys(mockOpsData).map(id => (
            <button
              key={id}
              onClick={() => setSelectedOpId(id)}
              className={`w-full flex items-center justify-between px-4 py-3 rounded-lg transition-all ${
                selectedOpId === id ? 'bg-blue-600 text-white' : 'hover:bg-slate-800 text-slate-400'
              }`}
            >
              {id} <ChevronRight size={14} />
            </button>
          ))}
        </div>
      </aside>

      {/* Main */}
      <main className="flex-1 p-10 overflow-y-auto">
        <header className="mb-8">
          <span className="text-blue-400 font-mono text-sm">{data.category}</span>
          <h2 className="text-4xl font-bold">{data.id} Analysis</h2>
        </header>

        <div className="grid grid-cols-3 gap-6 mb-8">
          <div className="bg-[#1e293b] p-6 rounded-xl border border-slate-700">
            <p className="text-slate-500 text-sm mb-1 flex items-center gap-2"><Activity size={14}/> Throughput</p>
            <p className="text-2xl font-bold text-emerald-400">{data.performance.throughput}</p>
          </div>
          <div className="bg-[#1e293b] p-6 rounded-xl border border-slate-700">
            <p className="text-slate-500 text-sm mb-1 flex items-center gap-2"><Database size={14}/> Occupancy</p>
            <p className="text-2xl font-bold text-blue-400">{data.performance.occupancy}</p>
          </div>
          <div className="bg-[#1e293b] p-6 rounded-xl border border-slate-700">
            <p className="text-slate-500 text-sm mb-1 flex items-center gap-2"><Code size={14}/> Status</p>
            <p className="text-2xl font-bold text-white">Verified</p>
          </div>
        </div>

        <div className="grid grid-cols-2 gap-8">
          <div className="bg-[#1e293b] p-6 rounded-xl border border-slate-700 h-80">
            <h3 className="text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">Latency Comparison</h3>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#334155" vertical={false} />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={12} />
                <YAxis stroke="#94a3b8" fontSize={12} />
                <Tooltip cursor={{fill: '#2d3748'}} contentStyle={{backgroundColor: '#1e293b', border: 'none'}} />
                <Bar dataKey="value" radius={[4, 4, 0, 0]}>
                  {latencyData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="bg-[#1e293b] p-6 rounded-xl border border-slate-700">
            <h3 className="text-sm font-semibold text-slate-400 mb-6 uppercase tracking-wider">Kernel Snippet</h3>
            <pre className="bg-[#0f172a] p-4 rounded-lg text-sm font-mono text-blue-300 overflow-x-auto h-56 leading-relaxed">
              <code>{data.cudaCode}</code>
            </pre>
          </div>
        </div>
      </main>
    </div>
  );
}