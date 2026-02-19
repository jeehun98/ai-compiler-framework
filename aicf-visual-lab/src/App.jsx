import React, { useMemo, useState } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import {
  BarChart, Bar, XAxis, YAxis, Tooltip,
  ResponsiveContainer, Cell
} from 'recharts';
import {
  Cpu, ChevronRight, Zap, Share2, Layers, ShieldCheck, Activity,
  Terminal, Scale, Eye, Focus, History, Boxes
} from 'lucide-react';

import { allOpsData } from './data/index.js';
import KernelDeepDive from './components/KernelDeepDive';

export default function App() {
  const [selectedOpId, setSelectedOpId] = useState('AdamStep');
  const [isModalOpen, setIsModalOpen] = useState(false);
  const data = allOpsData[selectedOpId];

  if (!data) return (
    <div className="p-10 text-blue-300 bg-[#0f172a] h-screen flex flex-col items-center justify-center font-mono italic">
      <div className="animate-pulse mb-4 text-2xl">AICF Engine Analyzing...</div>
      <div className="text-slate-500 text-sm italic">그래프 의미론 분석 및 최적화 경로 탐색 중</div>
    </div>
  );

  const semantic = data.semantics ?? data.semantic ?? null;
  const formula = data.canonical?.formula ?? '';
  const shapes = data.canonical?.shapes ?? {};
  const interpretation = data.canonical?.interpretation ?? {};
  const latency = data.performance?.latency ?? {};

  const latencyData = useMemo(() => ([
    { name: 'PyTorch (기본)', value: latency.pytorch ?? 0, color: '#64748b' },
    { name: 'torch.compile', value: latency.torch_compile ?? 0, color: '#94a3b8' },
    { name: 'AICF 최적화', value: latency.ours ?? 0, color: '#3b82f6' },
  ]), [latency.pytorch, latency.torch_compile, latency.ours]);

  const km = data.kernel?.metrics ?? {};
  const chosenVariant = data.lowering?.chosen?.variant ?? 'Standard_Kernel';
  const hasDeepDive = !!(data.kernel_evolution || data.evolution);

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 font-sans overflow-hidden italic-vars">
      {/* 사이드바 */}
      <aside className="w-80 bg-[#1e293b] border-r border-slate-700 p-6 flex flex-col shadow-2xl z-10">
        <h1 className="text-xl font-bold text-blue-400 mb-10 tracking-tight flex items-center gap-2 uppercase">
          <Cpu size={24} className="text-blue-500" /> AICF Lab <span className="text-[10px] text-slate-500 font-normal">v1.0</span>
        </h1>
        <div className="space-y-2 flex-1 overflow-y-auto pr-2 scrollbar-thin scrollbar-thumb-slate-700">
          <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest mb-4 px-4 font-mono">Graph Trace</p>
          {Object.keys(allOpsData).map(id => (
            <button
              key={id}
              onClick={() => setSelectedOpId(id)}
              className={`w-full flex flex-col items-start px-5 py-4 rounded-2xl transition-all duration-300 ${
                selectedOpId === id ? 'bg-blue-600 text-white shadow-lg scale-[1.02]' : 'hover:bg-slate-800 text-slate-400 opacity-70 hover:opacity-100'
              }`}
            >
              <div className="flex justify-between w-full items-center uppercase font-black text-xs tracking-widest">
                <span>{id}</span>
                <ChevronRight size={14} opacity={selectedOpId === id ? 1 : 0.3} />
              </div>
              <span className={`text-[10px] mt-1 ${selectedOpId === id ? 'text-blue-100' : 'text-slate-500'}`}>
                {allOpsData[id]?.category ?? '연산자 분류'}
              </span>
            </button>
          ))}
        </div>
      </aside>

      {/* 메인 콘텐츠 */}
      <main className="flex-1 p-10 overflow-y-auto space-y-12 bg-gradient-to-b from-[#0f172a] to-[#1e293b]/20">
        <header className="flex justify-between items-end border-b border-slate-800 pb-8">
          <div>
            <span className="text-blue-500 font-mono text-xs uppercase tracking-[0.4em] font-black italic">Architecture Trace Report</span>
            <h2 className="text-6xl font-black mt-2 tracking-tighter italic">
              {data.id} <span className="text-slate-700 font-light not-italic text-4xl ml-2 text-blue-400/30">Trace</span>
            </h2>
          </div>
          <div className="flex items-center gap-2 text-emerald-400 font-black bg-emerald-400/5 px-4 py-2 rounded-xl border border-emerald-400/10 text-xs uppercase tracking-widest shadow-lg">
            <ShieldCheck size={16} /> 수학적 불변성 확정 (Semantic Anchored)
          </div>
        </header>

        {/* 1. 연산 본질 섹션 */}
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-blue-500">
            <Share2 size={28} />
            <h3 className="text-3xl font-black uppercase tracking-tighter italic">1. 연산 본질 및 데이터 정의</h3>
          </div>
          <p className="text-slate-500 text-sm ml-10 -mt-6 italic leading-relaxed max-w-3xl">
            {data.descriptions?.essence ?? "해당 연산의 수학적/의미론적 본질을 분석합니다."}
          </p>

          <div className="grid grid-cols-12 gap-6">
            <div className="col-span-12 lg:col-span-8 bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
              <div className="bg-[#0b1120] p-12 rounded-3xl border border-slate-800/50 w-full text-center shadow-inner mb-8">
                <div className="text-5xl text-blue-400 drop-shadow-[0_0_15px_rgba(96,165,250,0.3)] overflow-x-auto scrollbar-hide">
                  <BlockMath math={formula} />
                </div>
                <div className="mt-8 flex justify-center gap-8 text-slate-500 font-mono text-xs">
                  {Object.entries(shapes).map(([tensor, shape]) => (
                    <div key={tensor} className="flex gap-2 items-center bg-[#0f172a] px-4 py-2 rounded-xl border border-slate-800">
                      <Boxes size={14} className="text-blue-500/40" />
                      <span className="text-blue-400 font-bold">{tensor}:</span>
                      <span className="italic">{shape}</span>
                    </div>
                  ))}
                </div>
              </div>

              <div className="space-y-4">
                <p className="text-[10px] text-slate-500 uppercase font-black tracking-[0.2em] border-l-2 border-blue-500 pl-3 mb-6">데이터 흐름 및 축별 의미</p>
                <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                  {Object.keys(semantic?.axes ?? {}).map((axisKey) => (
                    <div key={axisKey} className="relative p-6 bg-[#0f172a] rounded-2xl border border-slate-800 group hover:border-blue-500/40 transition-all shadow-lg overflow-hidden">
                      <div className="absolute -bottom-2 -right-2 text-6xl font-black text-blue-500/5 group-hover:text-blue-500/10 transition-colors uppercase italic font-mono pointer-events-none">{axisKey}</div>
                      <div className="mb-4 relative z-10">
                        <p className="text-blue-500 font-black text-[10px] uppercase tracking-widest mb-1">{semantic.axes[axisKey].name}</p>
                        <p className="text-sm font-bold text-slate-200 leading-snug">{interpretation[axisKey] || "정의되지 않음"}</p>
                      </div>
                      <div className="pt-3 border-t border-slate-800/50 relative z-10">
                        <p className="text-[10px] text-slate-500"><span className="text-slate-400 font-black uppercase mr-1">Role:</span>"{semantic.axes[axisKey].role}"</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            <div className="col-span-12 lg:col-span-4 space-y-4">
              <p className="text-[10px] text-slate-500 uppercase font-black tracking-[0.2em] border-l-2 border-blue-500 pl-3 mb-4 font-mono">Optimization Constraints</p>
              {semantic?.invariants?.map(inv => (
                <div key={inv.id} className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 hover:border-blue-500/50 transition-all shadow-xl group">
                  <div className="flex justify-between items-start mb-4">
                    <p className="text-sm font-black text-blue-400 uppercase tracking-tight italic">{inv.name}</p>
                    <ShieldCheck size={14} className="text-emerald-500/50" />
                  </div>
                  <div className="bg-[#0f172a] px-3 py-2 rounded-xl border border-slate-800 mb-3 overflow-x-auto scrollbar-hide">
                    <div className="text-xs text-blue-200/70 italic font-mono"><InlineMath math={inv.metric} /></div>
                  </div>
                  <div className="bg-emerald-500/5 px-3 py-3 rounded-xl border border-emerald-500/20 text-[11px] text-emerald-400 font-bold">{inv.threshold}</div>
                </div>
              ))}
            </div>
          </div>
        </section>

        {/* 2. 연쇄 최적화 전략 섹션 */}
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-purple-400">
            <Eye size={28} />
            <h3 className="text-3xl font-black uppercase tracking-tighter italic">2. 연쇄 최적화 전략</h3>
          </div>
          <p className="text-slate-500 text-sm ml-10 -mt-6 italic leading-relaxed max-w-3xl">
            {data.descriptions?.strategy ?? "인접한 후행 연산의 특성을 분석하여 최적화 경로를 탐색합니다."}
          </p>
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {semantic?.sensitivity?.downstream?.map((ds, i) => (
              <div key={i} className="bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 flex gap-8 items-start shadow-xl group relative overflow-hidden transition-all hover:border-purple-500/40">
                <div className="bg-purple-500/10 p-5 rounded-2xl border border-purple-500/20 text-purple-400 shrink-0"><Focus size={32} /></div>
                <div className="flex-1 space-y-5">
                  <h4 className="text-2xl font-black text-white italic uppercase tracking-tighter">{ds.name}</h4>
                  <div className="relative">
                    <p className="text-[9px] text-slate-600 font-black uppercase tracking-widest mb-1.5 ml-1">Detection Rule</p>
                    <div className="text-sm text-slate-200 leading-relaxed bg-[#0f172a] p-4 rounded-2xl border border-slate-800 italic shadow-inner overflow-x-auto scrollbar-hide">
                      <div className="text-[11px] whitespace-normal break-keep"><InlineMath math={ds.rule} /></div>
                    </div>
                  </div>
                  <div className="flex items-center gap-3 bg-emerald-500/5 px-4 py-3 rounded-2xl border border-emerald-500/10"><Zap size={14} className="text-emerald-400" /><span className="text-xs font-bold text-emerald-400">{ds.hint}</span></div>
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* 3. 하드웨어 매핑 섹션 */}
        <section className="space-y-8 pb-24">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3 text-emerald-400"><Zap size={28} /><h3 className="text-3xl font-black uppercase tracking-tighter italic">3. 하드웨어 매핑 및 최적화 구현</h3></div>
            {hasDeepDive && <button onClick={() => setIsModalOpen(true)} className="flex items-center gap-2 px-6 py-3 bg-emerald-600/10 hover:bg-emerald-600 border border-emerald-500/30 text-emerald-400 hover:text-white rounded-2xl font-black text-xs uppercase transition-all shadow-lg"><History size={16} />최적화 히스토리 보기</button>}
          </div>
          <p className="text-slate-500 text-sm ml-10 -mt-6 italic leading-relaxed max-w-3xl">{data.descriptions?.hardware ?? "GPU 하드웨어 매핑 전략을 수행합니다."}</p>
          <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
            <div className="lg:col-span-5 bg-[#1e293b] p-10 rounded-[2.5rem] border border-slate-800 shadow-2xl flex flex-col">
              <div className="flex items-center gap-2 mb-8 text-emerald-400"><Terminal size={20} /><h4 className="text-[10px] font-black uppercase tracking-[0.2em]">Lowering Decision Engine</h4></div>
              <div className="space-y-10">
                <div><p className="text-[10px] text-slate-500 uppercase font-black mb-3 ml-2">Selected Variant</p><p className="text-2xl font-black text-white italic ml-2">"{chosenVariant}"</p></div>
                <div className="space-y-4 text-sm">
                  {data.lowering?.chosen?.reason?.map((r, i) => (
                    <div key={i} className="flex gap-4 p-4 bg-[#0f172a] rounded-2xl border border-slate-800 text-slate-400 leading-relaxed font-bold border-l-4 border-l-emerald-600/50">
                      <span className="text-emerald-500 font-mono text-xs mt-1 shrink-0">0{i + 1}</span>
                      <div className="flex-1 overflow-x-auto scrollbar-hide">
                        <div className="text-[11px] whitespace-normal break-keep"><InlineMath math={r} /></div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
            <div className="lg:col-span-7 grid grid-cols-2 gap-6">
              <MetricCard title="최대 처리량" value={km.throughput} color="text-emerald-400" icon={<Activity size={16} />} />
              <MetricCard title="메모리 재사용률" value={km.memory_reuse} color="text-purple-400" icon={<Layers size={16} />} />
              <div className="col-span-2 bg-[#1e293b] p-8 rounded-3xl border border-slate-700 shadow-xl">
                <div className="flex items-center justify-between mb-8"><div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] font-black uppercase"><Scale size={18} /> Semantic Cost Model</div><div className="text-xs font-mono text-blue-400 italic"><InlineMath math={data.costModel?.semanticLoss || ''} /></div></div>
                <div className="grid grid-cols-3 gap-4">
                  {Object.entries(data.costModel?.weights_hint?.default ?? {}).map(([k, v]) => (
                    <div key={k} className="flex flex-col items-center gap-2 p-4 bg-[#0f172a]/50 rounded-2xl border border-slate-800"><div className="text-lg font-black text-slate-100">{v}</div><p className="text-[9px] text-slate-600 uppercase font-black tracking-tighter">{k}</p></div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* 성능 차트 */}
        <section className="bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-xl pb-12">
          <h4 className="text-slate-500 text-[10px] font-black mb-12 uppercase text-center italic font-mono">Physical Performance Comparison (Latency ms)</h4>
          <div className="h-64">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={latencyData} layout="vertical">
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={11} width={100} />
                <Tooltip cursor={{ fill: '#2d3748' }} contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: '12px' }} />
                <Bar dataKey="value" barSize={32} radius={[0, 8, 8, 0]}>
                  {latencyData.map((e, i) => <Cell key={i} fill={e.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </section>
      </main>

      <KernelDeepDive isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} data={data} />

      {/* 스타일 보강: 스크롤바 숨김 */}
      <style jsx="true">{`
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  );
}

function MetricCard({ title, value, color, icon }) {
  return (
    <div className="bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-lg group transition-all hover:border-emerald-500/30">
      <div className="flex items-center gap-3 text-slate-500 mb-4 group-hover:text-emerald-400 transition-colors">
        {icon}
        <p className="text-[10px] uppercase font-black tracking-[0.2em]">{title}</p>
      </div>
      <p className={`text-4xl font-black font-mono tracking-tighter ${color}`}>{value ?? '—'}</p>
    </div>
  );
}