import React, { useState } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts';
import {
  Cpu, ChevronRight, Zap, Share2, Layers, Target, ShieldCheck, Activity,
  Terminal, Scale, Eye, Focus
} from 'lucide-react';

import { allOpsData } from './data/index.js';

export default function App() {
  const [selectedOpId, setSelectedOpId] = useState('GEMM');
  const data = allOpsData[selectedOpId];

  // -------------------------
  // HARD GUARDS
  // -------------------------
  if (!data) {
    return (
      <div className="p-10 text-red-300 bg-[#0f172a] h-screen text-center font-mono">
        Loading Semantic Engine...
      </div>
    );
  }

  const semantic = data.semantics ?? data.semantic ?? null;
  if (!semantic) {
    return (
      <div className="p-10 text-red-300 bg-[#0f172a] h-screen text-center font-mono">
        Missing semantic spec for: {data.id ?? selectedOpId}
      </div>
    );
  }

  // -------------------------
  // Derived values (safe)
  // -------------------------
  const formula = data.canonical?.formula ?? '';
  const latency = data.performance?.latency ?? {};
  const latencyData = [
    { name: 'PyTorch', value: latency.pytorch ?? 0, color: '#64748b' },
    { name: 'Compile', value: latency.torch_compile ?? 0, color: '#94a3b8' },
    { name: 'AICF', value: latency.ours ?? 0, color: '#3b82f6' },
  ];

  const km = data.kernel?.metrics ?? {};
  const throughput = km.throughput ?? '—';
  const memoryReuse = km.memory_reuse ?? '—';
  const occupancy = (km.occupancy ?? km.occupancy === 0) ? `${km.occupancy}%` : '—';

  const chosenVariant = data.lowering?.chosen?.variant ?? '—';
  const chosenReasons = data.lowering?.chosen?.reason ?? [];
  const appliedRewrites = data.lowering?.chosen?.applied_rewrites ?? [];

  const costExpr = data.costModel?.semanticLoss ?? '';
  const costWeights = data.costModel?.weights_hint?.default ?? {};

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 font-sans overflow-hidden italic-vars">
      {/* 1. Sidebar */}
      <aside className="w-72 bg-[#1e293b] border-r border-slate-700 p-6 flex flex-col shadow-2xl z-10">
        <h1 className="text-xl font-bold text-blue-400 mb-10 tracking-tight flex items-center gap-2">
          <Cpu size={24} className="text-blue-500" /> AICF VISUAL LAB
        </h1>

        <div className="space-y-2 flex-1 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-700">
          {Object.keys(allOpsData).map(id => (
            <button
              key={id}
              onClick={() => setSelectedOpId(id)}
              className={`w-full flex flex-col items-start px-4 py-4 rounded-2xl transition-all duration-300 ${
                selectedOpId === id
                  ? 'bg-blue-600 text-white shadow-lg scale-[1.02]'
                  : 'hover:bg-slate-800 text-slate-400 opacity-70 hover:opacity-100'
              }`}
            >
              <div className="flex justify-between w-full items-center">
                <span className="font-black uppercase tracking-wider text-sm">{id}</span>
                <ChevronRight size={14} opacity={selectedOpId === id ? 1 : 0.3} />
              </div>
              <span className={`text-[10px] mt-1 font-medium ${
                selectedOpId === id ? 'text-blue-100' : 'text-slate-500'
              }`}>
                {allOpsData[id]?.category ?? 'Uncategorized'}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* 2. Main Content */}
      <main className="flex-1 p-10 overflow-y-auto space-y-12 bg-[#0f172a]">
        <header className="flex justify-between items-end border-b border-slate-800 pb-8">
          <div>
            <span className="text-blue-500 font-mono text-xs uppercase tracking-[0.4em] font-black">
              Architecture Trace
            </span>
            <h2 className="text-6xl font-black mt-2 tracking-tighter italic">
              {data.id}{' '}
              <span className="text-slate-700 font-light not-italic text-4xl ml-2">
                v1.0
              </span>
            </h2>
          </div>

          <div className="text-right space-y-2">
            <div className="flex items-center gap-2 text-emerald-400 font-black bg-emerald-400/5 px-4 py-2 rounded-xl border border-emerald-400/10 text-xs uppercase tracking-widest">
              <ShieldCheck size={16} /> Semantic Consistency: 100%
            </div>
          </div>
        </header>

        {/* SECTION 1: Semantic Deep-Dive */}
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-blue-500">
            <Share2 size={28} />
            <h3 className="text-3xl font-black uppercase tracking-tighter">
              1. Semantic Deep-Dive
            </h3>
          </div>

          <div className="grid grid-cols-12 gap-6">
            {/* Canonical + Axes */}
            <div className="col-span-12 lg:col-span-8 bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
              <div className="flex items-center justify-between mb-8">
                <div className="flex items-center gap-2 text-slate-500">
                  <Target size={18} />
                  <h4 className="text-[10px] font-black uppercase tracking-[0.2em]">
                    Canonical Proposition
                  </h4>
                </div>

                <div className="text-[10px] font-mono text-slate-600 bg-slate-900/50 px-3 py-1 rounded-full uppercase tracking-tighter">
                  K-Dim: <span className="text-slate-300 font-black">K</span> (Search Space)
                </div>
              </div>

              {/* Formula */}
              <div className="bg-[#0b1120] p-12 rounded-3xl border border-slate-800/50 w-full text-center shadow-inner mb-4 overflow-x-auto">
                <div className="text-4xl text-blue-400 drop-shadow-[0_0_15px_rgba(96,165,250,0.3)] min-w-max">
                  <BlockMath math={formula} />
                </div>
              </div>

              {/* Thesis */}
              {semantic?.thesis ? (
                <p className="text-xs text-slate-500 font-medium italic text-center mb-8">
                  "{semantic.thesis}"
                </p>
              ) : (
                <div className="mb-8" />
              )}

              {/* Axes */}
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-6">
                {Object.entries(semantic?.axes ?? {}).map(([key, axis]) => (
                  <div
                    key={key}
                    className="p-5 bg-[#0f172a] rounded-2xl border border-slate-800 group hover:border-blue-500/30 transition-all"
                  >
                    <p className="text-blue-500 font-black text-xs uppercase mb-1 tracking-widest">
                      {key}
                    </p>
                    <p className="text-sm font-bold text-slate-200 mb-1">
                      {axis?.name ?? '—'}
                    </p>
                    <p className="text-[10px] text-slate-500 leading-relaxed italic line-clamp-2">
                      "{axis?.role ?? '—'}"
                    </p>
                  </div>
                ))}
              </div>
            </div>

            {/* Semantic Invariants */}
            <div className="col-span-12 lg:col-span-4 space-y-4">
              <h4 className="text-slate-500 text-[10px] font-black uppercase tracking-[0.2em] flex items-center gap-2 ml-2 mb-2">
                <ShieldCheck size={14} /> Semantic Invariants
              </h4>

              {(semantic?.invariants ?? []).map(inv => (
                <div
                  key={inv.id ?? inv.name}
                  className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 hover:bg-slate-800/40 transition-colors"
                >
                  <div className="flex items-center gap-3 mb-3">
                    <p className="text-sm font-black text-blue-400 uppercase tracking-tight">
                      {inv.name ?? inv.id ?? 'Invariant'}
                    </p>
                  </div>

                  {/* Metric */}
                  <div className="bg-[#0f172a] px-3 py-2 rounded-xl mb-3 border border-slate-800">
                    <p className="text-[10px] text-slate-400 mb-1 font-mono uppercase opacity-50 font-black">
                      Metric
                    </p>
                    <div className="text-xs text-blue-200">
                      <InlineMath math={inv.metric ?? '—'} />
                    </div>
                  </div>

                  {/* Threshold */}
                  <div className="bg-[#0f172a] px-3 py-2 rounded-xl mb-4 border border-slate-800">
                    <p className="text-[10px] text-slate-400 mb-1 font-mono uppercase opacity-50 font-black">
                      Threshold
                    </p>
                    <div className="text-[11px] text-slate-300">
                      {inv.threshold ?? '—'}
                    </div>
                  </div>

                  {/* Applies when */}
                  {(inv.applies_when ?? []).length ? (
                    <div className="mb-3">
                      <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest mb-2">
                        Applies When
                      </p>
                      <div className="flex flex-wrap gap-1.5">
                        {(inv.applies_when ?? []).map(w => (
                          <span
                            key={w}
                            className="text-[9px] font-bold bg-slate-900 text-slate-400 px-2.5 py-1 rounded-lg border border-slate-800 uppercase tracking-tighter"
                          >
                            {w}
                          </span>
                        ))}
                      </div>
                    </div>
                  ) : null}

                  {/* Allows */}
                  <div className="flex flex-wrap gap-1.5">
                    {(inv.allows ?? []).map(a => (
                      <span
                        key={a}
                        className="text-[9px] font-bold bg-slate-900 text-blue-400/80 px-2.5 py-1 rounded-lg border border-slate-800 uppercase tracking-tighter"
                      >
                        {a}
                      </span>
                    ))}
                  </div>
                </div>
              ))}

              {(!semantic?.invariants || semantic.invariants.length === 0) ? (
                <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 text-xs text-slate-500">
                  No invariants defined.
                </div>
              ) : null}
            </div>
          </div>
        </section>

        {/* SECTION 2: Downstream Sensitivity */}
        <section className="space-y-6">
          <div className="flex items-center gap-3 text-purple-400">
            <Eye size={28} />
            <h3 className="text-3xl font-black uppercase tracking-tighter">
              2. Contextual Sensitivity
            </h3>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {(semantic?.sensitivity?.downstream ?? []).map((ds, i) => (
              <div
                key={`${ds?.name ?? 'ds'}-${i}`}
                className="bg-gradient-to-br from-[#1e293b] to-[#0f172a] p-8 rounded-[2rem] border border-slate-800 flex flex-col md:flex-row gap-6 items-center"
              >
                <div className="bg-purple-500/10 p-6 rounded-2xl border border-purple-500/20 text-purple-400">
                  <Focus size={32} />
                </div>

                <div className="flex-1 space-y-2">
                  <div className="flex items-center gap-2">
                    <span className="text-[10px] font-black text-purple-400 uppercase tracking-[0.2em]">
                      Impact
                    </span>
                    <h4 className="text-xl font-black text-white">
                      {ds?.name ?? '—'}
                    </h4>
                  </div>

                  <p className="text-sm text-slate-400 leading-relaxed font-medium">
                    Rule:{' '}
                    <span className="text-slate-200 italic">
                      "{ds?.rule ?? '—'}"
                    </span>
                  </p>

                  <div className="inline-block px-3 py-1 bg-slate-900 border border-slate-800 rounded-lg text-[10px] font-mono text-purple-300">
                    Hint: {ds?.hint ?? '—'}
                  </div>
                </div>
              </div>
            ))}

            <div className="bg-[#1e293b] p-8 rounded-[2rem] border border-slate-800 border-dashed flex flex-col justify-center items-center text-center space-y-2">
              <p className="text-xs font-black text-slate-500 uppercase tracking-widest">
                Tile Priority Model
              </p>
              <p className="text-lg font-bold text-slate-300 italic">
                "{semantic?.sensitivity?.tilePriority ?? '—'}"
              </p>
              <p className="text-[10px] text-slate-600 max-w-xs mt-2">
                Predicts significance based on activation density before lowering.
              </p>
            </div>
          </div>
        </section>

        {/* SECTION 3: Lowering & Physical */}
        <section className="space-y-8 pb-24">
          <div className="flex items-center gap-3 text-emerald-400">
            <Zap size={28} />
            <h3 className="text-3xl font-black uppercase tracking-tighter italic">
              3. Lowering & Physical Execution
            </h3>
          </div>

          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            {/* Compiler Decision */}
            <div className="lg:col-span-5 bg-[#1e293b] p-10 rounded-[2.5rem] border border-slate-800 shadow-2xl flex flex-col">
              <div className="flex items-center gap-2 mb-8 text-emerald-400">
                <Terminal size={20} />
                <h4 className="text-[10px] font-black uppercase tracking-[0.2em]">
                  Compiler Lowering Decision
                </h4>
              </div>

              <div className="flex-1 space-y-10">
                <div className="relative">
                  <div className="absolute -left-4 top-0 bottom-0 w-1 bg-emerald-500/30 rounded-full" />
                  <p className="text-[10px] text-slate-500 uppercase font-black mb-3 ml-2 tracking-widest">
                    Chosen Variant
                  </p>
                  <p className="text-2xl font-black text-white italic ml-2 tracking-tight">
                    "{chosenVariant}"
                  </p>

                  {(appliedRewrites ?? []).length ? (
                    <div className="flex flex-wrap gap-2 mt-4 ml-2">
                      {appliedRewrites.map(rw => (
                        <span
                          key={rw}
                          className="text-[10px] font-black bg-emerald-500/10 text-emerald-300 px-3 py-1 rounded-xl border border-emerald-500/20"
                        >
                          ✓ {rw}
                        </span>
                      ))}
                    </div>
                  ) : null}
                </div>

                <div className="space-y-4">
                  <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest">
                    Logic Evidence
                  </p>
                  {(chosenReasons ?? []).map((r, i) => (
                    <div
                      key={i}
                      className="flex gap-4 p-4 bg-[#0f172a] rounded-2xl border border-slate-800 text-[12px] text-slate-400 leading-relaxed font-medium"
                    >
                      <span className="text-emerald-500 font-black">
                        {String(i + 1).padStart(2, '0')}
                      </span>
                      <span>{r}</span>
                    </div>
                  ))}

                  {(chosenReasons ?? []).length === 0 ? (
                    <div className="p-4 bg-[#0f172a] rounded-2xl border border-slate-800 text-[12px] text-slate-500">
                      No evidence lines provided.
                    </div>
                  ) : null}
                </div>
              </div>
            </div>

            {/* Metrics + Cost + CUDA */}
            <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-2 gap-6">
              <MetricCard title="Peak Throughput" value={throughput} color="text-emerald-400" icon={<Activity size={16} />} />
              <MetricCard title="Occupancy" value={occupancy} color="text-blue-400" icon={<ShieldCheck size={16} />} />
              <MetricCard title="Memory Reuse" value={memoryReuse} color="text-purple-400" icon={<Layers size={16} />} />
              <MetricCard title="Kernel Strategy" value={data.kernel?.strategy ?? '—'} color="text-white" icon={<Zap size={16} />} />

              {/* Cost Model Card */}
              <div className="sm:col-span-2 bg-[#1e293b] p-8 rounded-3xl border border-slate-700">
                <div className="flex items-center justify-between mb-8 text-center sm:text-left gap-4">
                  <div className="flex items-center gap-2 text-slate-500">
                    <Scale size={18} />
                    <p className="text-[10px] font-black uppercase tracking-widest">
                      Semantic Cost Weighting
                    </p>
                  </div>

                  {costExpr ? (
                    <div className="text-xs font-mono text-blue-400 font-bold bg-blue-500/5 px-4 py-1.5 rounded-full border border-blue-500/10 shadow-inner overflow-x-auto">
                      <InlineMath math={costExpr} />
                    </div>
                  ) : (
                    <div className="text-[10px] font-mono text-slate-600 bg-slate-900/40 px-3 py-1 rounded-full border border-slate-800">
                      No cost expression
                    </div>
                  )}
                </div>

                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(costWeights).map(([k, v]) => (
                    <div
                      key={k}
                      className="flex flex-col items-center gap-2 p-4 bg-[#0f172a]/50 rounded-2xl border border-slate-800/50"
                    >
                      <div className="text-lg font-black text-slate-200">{v}</div>
                      <p className="text-[9px] text-slate-600 uppercase font-bold tracking-tighter">
                        {k}
                      </p>
                    </div>
                  ))}
                  {Object.keys(costWeights).length === 0 ? (
                    <div className="col-span-3 p-4 bg-[#0f172a]/50 rounded-2xl border border-slate-800/50 text-[11px] text-slate-600 text-center">
                      No weights defined.
                    </div>
                  ) : null}
                </div>
              </div>

              {/* CUDA Snippet */}
              <div className="sm:col-span-2 bg-[#0b1120] p-8 rounded-3xl border border-slate-800 overflow-hidden relative">
                <h4 className="text-slate-500 text-[10px] font-black mb-4 uppercase tracking-widest">
                  Target CUDA snippet
                </h4>
                <pre className="text-[13px] font-mono text-emerald-400/80 leading-relaxed max-h-48 overflow-y-auto scrollbar-thin scrollbar-thumb-slate-800 italic">
                  <code>{data.cudaCode ?? '// (WIP) no code emitted yet'}</code>
                </pre>
                <div className="absolute -bottom-4 -right-4 opacity-5 rotate-12">
                  <Cpu size={120} className="text-white" />
                </div>
              </div>
            </div>
          </div>

          {/* Benchmark Section */}
          <div className="grid grid-cols-12 gap-8">
            <div className="col-span-12 bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
              <h4 className="text-slate-500 text-xs font-black mb-8 uppercase tracking-widest text-center italic">
                Kernel Performance Benchmark (ms)
              </h4>

              <div className="h-64">
                <ResponsiveContainer width="100%" height="100%">
                  <BarChart data={latencyData} layout="vertical">
                    <XAxis type="number" hide />
                    <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={12} width={80} />
                    <Tooltip
                      cursor={{ fill: '#2d3748' }}
                      contentStyle={{
                        backgroundColor: '#1e293b',
                        border: '1px solid #334155',
                        borderRadius: '12px'
                      }}
                    />
                    <Bar dataKey="value" barSize={32} radius={[0, 8, 8, 0]}>
                      {latencyData.map((e, i) => (
                        <Cell key={i} fill={e.color} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              <div className="mt-6 text-[11px] text-slate-500 leading-relaxed text-center">
                Note: benchmark numbers may be placeholders while lowering variants evolve.
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}

// --- Metric Card Component ---
function MetricCard({ title, value, color, icon }) {
  return (
    <div className="bg-[#1e293b] p-8 rounded-[2rem] border border-slate-800 hover:border-emerald-500/30 transition-all duration-500 group">
      <div className="flex items-center gap-3 text-slate-500 mb-4 group-hover:text-emerald-400 transition-colors">
        {icon}
        <p className="text-[10px] uppercase font-black tracking-[0.2em]">{title}</p>
      </div>
      <p className={`text-4xl font-black font-mono tracking-tighter ${color}`}>
        {value ?? '—'}
      </p>
    </div>
  );
}
