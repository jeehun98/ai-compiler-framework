// src/pages/OpsPage.jsx
import React, { useMemo, useState, useEffect } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import {
  Cpu, ChevronRight, Zap, Share2, Layers, ShieldCheck, Activity,
  Terminal, Scale, Eye, Focus, History, Boxes, Menu, X, ArrowUpRight
} from 'lucide-react';

import { useSearchParams, Link } from 'react-router-dom';
import { allOpsData } from '../data/index.js';
import KernelDeepDive from '../components/KernelDeepDive.jsx';

export default function OpsPage() {
  const [searchParams, setSearchParams] = useSearchParams();

  const initialOp = searchParams.get('op') || 'AdamStep';
  const [selectedOpId, setSelectedOpId] = useState(initialOp);

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  // ✅ URL op 동기화 (뒤로가기/링크 이동 대응)
  useEffect(() => {
    const opFromUrl = searchParams.get('op');
    if (opFromUrl && opFromUrl !== selectedOpId) {
      setSelectedOpId(opFromUrl);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchParams]);

  const data = allOpsData[selectedOpId];

  // ✅ op 바뀌면 모바일 드로어 닫기
  useEffect(() => {
    setIsSidebarOpen(false);
  }, [selectedOpId]);

  const handleSelectOp = (id) => {
    setSelectedOpId(id);
    setSearchParams({ op: id }, { replace: true });
  };

  if (!data) {
    return (
      <div className="p-10 text-blue-400 bg-[#0f172a] min-h-screen flex flex-col items-center justify-center font-mono">
        <div className="animate-pulse mb-4 text-2xl font-black uppercase">AICF Engine Analyzing...</div>
        <div className="text-slate-500 text-sm">그래프 의미론 분석 및 최적화 경로 탐색 중</div>
      </div>
    );
  }

  const semantic = data.semantics ?? data.semantic ?? null;
  const formula = data.canonical?.formula ?? '';
  const shapes = data.canonical?.shapes ?? {};
  const interpretation = data.canonical?.interpretation ?? {};
  const latency = data.performance?.latency ?? {};

  const latencyData = useMemo(() => ([
    { name: 'PyTorch', value: latency.pytorch ?? 0, color: '#475569' },
    { name: 'torch.compile', value: latency.torch_compile ?? 0, color: '#64748b' },
    { name: 'AICF Optimized', value: latency.ours ?? 0, color: '#3b82f6' },
  ]), [latency.pytorch, latency.torch_compile, latency.ours]);

  const km = data.kernel?.metrics ?? {};
  const chosenVariant = data.lowering?.chosen?.variant ?? 'Standard_Kernel';
  const hasDeepDive = !!(data.kernel_evolution || data.evolution);

  return (
    // ✅ min-h-dvh로 모바일 주소창/세이프에어리어 흔들림 대응
    // ✅ x축 잘림 방지
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">

      {/* Mobile Top Header */}
      <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
        <div className="flex items-center justify-between px-5 py-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="bg-blue-600 p-2 rounded-xl">
              <Cpu size={18} className="text-white" />
            </div>
            <div className="leading-none">
              <div className="font-black text-blue-400 tracking-tight">AICF LAB</div>
              <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">v1.0.4 Stable</div>
            </div>
          </Link>

          <button
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200 active:scale-95 transition"
            aria-label="Open sidebar"
          >
            <Menu size={18} />
          </button>
        </div>
      </header>

      {/* Mobile Overlay */}
      {isSidebarOpen && (
        <div
          className="md:hidden fixed inset-0 z-40 bg-black/50"
          onClick={() => setIsSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* GLOBAL SIDEBAR */}
      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-50 md:z-10
          w-[85vw] max-w-[320px] md:w-80
          bg-[#1e293b] border-r border-slate-800
          flex flex-col shadow-2xl transition-transform duration-300
          ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
        role="dialog"
        aria-modal={isSidebarOpen ? 'true' : 'false'}
      >
        {/* Logo Area */}
        <div className="p-6 border-b border-slate-800 bg-[#0f172a]/50 flex items-center justify-between gap-3">
          <Link to="/" className="flex items-center gap-3 group min-w-0">
            <div className="bg-blue-600 p-2 rounded-xl group-hover:bg-blue-500 transition">
              <Cpu size={20} className="text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg font-black tracking-tight text-white leading-none truncate">AICF LAB</h1>
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                v1.0.4 Stable
              </span>
            </div>
          </Link>

          <button
            className="md:hidden p-2 rounded-xl border border-slate-700 bg-[#0f172a] text-slate-300"
            onClick={() => setIsSidebarOpen(false)}
            aria-label="Close sidebar"
          >
            <X size={16} />
          </button>
        </div>

        {/* Global Menu Navigation */}
        <nav className="p-4 border-b border-slate-800 space-y-1">
          <p className="px-3 text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">
            Navigation
          </p>

          <Link
            to="/"
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800 hover:text-white transition font-bold text-sm"
          >
            <Layers size={18} /> Dashboard
          </Link>

          <Link
            to="/ops"
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-blue-600/10 text-blue-400 font-bold text-sm border border-blue-500/20"
          >
            <Terminal size={18} /> Ops Explorer
          </Link>
        </nav>

        {/* Op List (Scrollable) */}
        <div className="flex-1 overflow-y-auto p-4 space-y-1 scrollbar-thin scrollbar-thumb-slate-700">
          <p className="px-3 text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">
            Available Operators
          </p>

          {Object.keys(allOpsData).map((id) => (
            <button
              key={id}
              onClick={() => handleSelectOp(id)}
              className={`w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm ${
                selectedOpId === id
                  ? 'bg-blue-600 text-white shadow-lg'
                  : 'text-slate-400 hover:bg-slate-800 hover:text-slate-200'
              }`}
            >
              <div className="min-w-0 flex flex-col items-start text-left">
                <span className="truncate w-full">{id}</span>
                <span className={`text-[10px] mt-0.5 truncate w-full ${
                  selectedOpId === id ? 'text-blue-100/90' : 'text-slate-500'
                }`}>
                  {allOpsData[id]?.category ?? '연산자 분류'}
                </span>
              </div>
              {selectedOpId === id ? <ArrowUpRight size={14} /> : <ChevronRight size={14} opacity={0.25} />}
            </button>
          ))}
        </div>
      </aside>

      {/* MAIN CONTENT */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* ✅ 모바일 상단바 공간 확보 */}
        <div className="md:hidden h-[68px]" />

        {/* ✅ 메인만 스크롤 (사이드바/오버레이 충돌 방지) */}
        <div className="flex-1 overflow-y-auto p-5 sm:p-8 lg:p-10 space-y-12 pb-32 bg-gradient-to-b from-[#0f172a] to-[#1e293b]/20">

          {/* Header Title Section */}
          <section className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 border-b border-slate-800 pb-8">
            <div className="space-y-2 min-w-0">
              <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black uppercase tracking-[0.3em]">
                <Activity size={14} /> Architecture Trace Report
              </div>
              <h2 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight break-words">
                {data.id} <span className="text-blue-500/30 font-light ml-2">Explorer</span>
              </h2>
            </div>

            <div className="flex items-center gap-2 text-emerald-400 font-bold bg-emerald-400/5 px-4 py-2 rounded-xl border border-emerald-400/10 text-[11px] uppercase tracking-widest w-fit">
              <ShieldCheck size={16} /> Semantic Anchored
            </div>
          </section>

          {/* 1. Essence Section */}
          <section className="space-y-6">
            <div className="flex items-center gap-3 text-blue-400">
              <Share2 size={24} />
              <h3 className="text-2xl font-black uppercase tracking-tight">
                1. 연산 본질 및 데이터 정의
              </h3>
            </div>

            <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
              {data.descriptions?.essence ?? "해당 연산의 수학적/의미론적 본질을 분석합니다."}
            </p>

            <div className="grid grid-cols-12 gap-6">
              <div className="col-span-12 lg:col-span-8 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                {/* Formula */}
                <div className="bg-[#0b1120] p-6 sm:p-10 rounded-3xl border border-slate-800/50 mb-8 overflow-x-auto scrollbar-hide">
                  <div className="text-3xl sm:text-4xl text-blue-400 text-center min-w-max">
                    <BlockMath math={formula} />
                  </div>

                  {/* Shapes (원본에 있던 배지 복원) */}
                  <div className="mt-6 flex flex-wrap justify-center gap-3 text-slate-500 font-mono text-xs">
                    {Object.entries(shapes).map(([tensor, shape]) => (
                      <div
                        key={tensor}
                        className="flex gap-2 items-center bg-[#0f172a] px-4 py-2 rounded-xl border border-slate-800 max-w-full"
                      >
                        <Boxes size={14} className="text-blue-500/40" />
                        <span className="text-blue-400 font-bold">{tensor}:</span>
                        <span className="italic break-all">{shape}</span>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Axes */}
                <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">
                  데이터 흐름 및 축별 의미
                </p>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {Object.keys(semantic?.axes ?? {}).map((axis) => (
                    <div
                      key={axis}
                      className="bg-[#0f172a] p-5 rounded-2xl border border-slate-800 relative group overflow-hidden hover:border-blue-500/30 transition"
                    >
                      <div className="absolute -right-2 -bottom-2 text-4xl font-black text-white/5 uppercase italic">
                        {axis}
                      </div>

                      <div className="relative z-10 text-[10px] font-black text-blue-500 uppercase tracking-widest mb-1">
                        {semantic.axes[axis].name}
                      </div>

                      <div className="relative z-10 text-sm font-bold text-slate-200 break-words">
                        {interpretation[axis] || "정의되지 않음"}
                      </div>

                      {/* Role (원본에 있던 role 표시 복원) */}
                      <div className="relative z-10 mt-4 pt-3 border-t border-slate-800/60 text-[10px] text-slate-500">
                        <span className="text-slate-400 font-black uppercase mr-1">Role:</span>
                        "{semantic.axes[axis].role}"
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Invariants */}
              <div className="col-span-12 lg:col-span-4 space-y-4">
                <div className="flex items-center justify-between px-2">
                  <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                    Optimization Constraints
                  </p>
                  <span className="text-[9px] text-emerald-500 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                    VERIFIED
                  </span>
                </div>

                {semantic?.invariants?.map((inv) => (
                  <div
                    key={inv.id}
                    className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 hover:border-blue-500/30 transition shadow-lg relative overflow-hidden"
                  >
                    <div className="absolute top-2 right-4 text-[8px] font-mono text-slate-700 opacity-60 uppercase tracking-tighter">
                      {inv.id}
                    </div>

                    <div className="text-blue-400 font-black text-xs uppercase tracking-tight mb-3">
                      {inv.name}
                    </div>

                    <div className="bg-[#0f172a] p-3 rounded-xl border border-slate-800 font-mono text-[11px] text-blue-200/70 mb-3 italic overflow-x-auto scrollbar-hide">
                      <InlineMath math={inv.metric} />
                    </div>

                    {/* Threshold + Allow Range (원본 정보 유지) */}
                    <div className="grid grid-cols-2 gap-2 mb-3">
                      <div className="bg-emerald-500/5 px-3 py-2 rounded-xl border border-emerald-500/10">
                        <p className="text-[8px] text-emerald-600 font-black uppercase tracking-widest mb-1 font-mono text-center">
                          Threshold
                        </p>
                        <div className="text-[10px] text-emerald-400 font-bold text-center leading-none">
                          <InlineMath math={inv.threshold} />
                        </div>
                      </div>

                      <div className="bg-blue-500/5 px-3 py-2 rounded-xl border border-blue-500/10">
                        <p className="text-[8px] text-blue-600 font-black uppercase tracking-widest mb-1 font-mono text-center">
                          Allow Range
                        </p>
                        <div className="text-[10px] text-blue-300 font-bold text-center leading-none">
                          {inv.allows?.length ?? 0} Strategies
                        </div>
                      </div>
                    </div>

                    <div className="flex flex-wrap gap-1">
                      {inv.allows?.map((a) => (
                        <span
                          key={a}
                          className="text-[8px] font-bold bg-slate-900 text-slate-500 px-2 py-0.5 rounded border border-slate-800 uppercase tracking-tighter hover:text-blue-400/80 transition-colors"
                        >
                          + {a}
                        </span>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </section>

          {/* 2. Strategy Section */}
          <section className="space-y-6">
            <div className="flex items-center gap-3 text-purple-400">
              <Eye size={24} />
              <h3 className="text-2xl font-black uppercase tracking-tight">
                2. 연쇄 최적화 전략
              </h3>
            </div>

            <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
              {data.descriptions?.strategy ?? "인접한 후행 연산의 특성을 분석하여 최적화 경로를 탐색합니다."}
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {semantic?.sensitivity?.downstream?.map((ds, i) => (
                <div
                  key={i}
                  className="bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl flex gap-6 items-start group hover:border-purple-500/30 transition"
                >
                  <div className="bg-purple-500/10 p-4 rounded-2xl border border-purple-500/20 text-purple-400 shrink-0">
                    <Focus size={24} />
                  </div>

                  <div className="space-y-4 flex-1 min-w-0">
                    <h4 className="text-xl font-black text-white uppercase tracking-tight break-words">
                      {ds.name}
                    </h4>

                    <div className="bg-[#0f172a] p-4 rounded-2xl border border-slate-800 font-mono text-xs text-slate-300 overflow-x-auto scrollbar-hide">
                      <InlineMath math={ds.rule} />
                    </div>

                    <div className="flex items-center gap-2 text-xs font-bold text-emerald-400 bg-emerald-400/5 px-3 py-2 rounded-lg border border-emerald-400/10">
                      <Zap size={14} /> <span className="break-words">{ds.hint}</span>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* 3. Hardware Mapping */}
          <section className="space-y-6">
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
              <div className="flex items-center gap-3 text-emerald-400">
                <Zap size={24} />
                <h3 className="text-2xl font-black uppercase tracking-tight">
                  3. 하드웨어 매핑 및 최적화 구현
                </h3>
              </div>

              {hasDeepDive && (
                <button
                  onClick={() => setIsModalOpen(true)}
                  className="w-fit flex items-center gap-2 px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-bold uppercase tracking-widest border border-slate-700 transition"
                >
                  <History size={14} />
                  History
                </button>
              )}
            </div>

            <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
              {data.descriptions?.hardware ?? "GPU 하드웨어 매핑 전략을 수행합니다."}
            </p>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
              {/* Lowering Decision */}
              <div className="lg:col-span-5 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                <div className="flex items-center gap-2 mb-6 text-emerald-400 font-black text-[10px] uppercase tracking-widest">
                  <Terminal size={16} /> Lowering Decision Engine
                </div>

                <p className="text-[10px] text-slate-500 uppercase font-black mb-2">
                  Selected Variant
                </p>
                <div className="text-2xl font-black text-white mb-6 break-words">
                  "{chosenVariant}"
                </div>

                <div className="space-y-3">
                  {data.lowering?.chosen?.reason?.map((r, i) => (
                    <div
                      key={i}
                      className="bg-[#0f172a] p-4 rounded-2xl border border-slate-800 text-sm font-bold text-slate-400 flex gap-3"
                    >
                      <span className="text-emerald-500 font-mono shrink-0">
                        0{i + 1}
                      </span>
                      <div className="min-w-0 overflow-x-auto scrollbar-hide">
                        <InlineMath math={r} />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {/* Metrics + CostModel + Chart */}
              <div className="lg:col-span-7 flex flex-col gap-6 min-w-0">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-6">
                  <MetricCard
                    title="Throughput"
                    value={km.throughput}
                    color="text-emerald-400"
                    icon={<Activity size={16} />}
                  />
                  <MetricCard
                    title="Mem Reuse"
                    value={km.memory_reuse}
                    color="text-purple-400"
                    icon={<Layers size={16} />}
                  />
                </div>

                {/* Cost Model (원본의 costModel 섹션 복원) */}
                <div className="bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3 mb-6">
                    <div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] font-black uppercase">
                      <Scale size={18} /> Semantic Cost Model
                    </div>
                    <div className="text-xs font-mono text-blue-400 italic break-words">
                      <InlineMath math={data.costModel?.semanticLoss || ''} />
                    </div>
                  </div>

                  <div className="grid grid-cols-3 gap-4">
                    {Object.entries(data.costModel?.weights_hint?.default ?? {}).map(([k, v]) => (
                      <div
                        key={k}
                        className="flex flex-col items-center gap-2 p-4 bg-[#0f172a]/50 rounded-2xl border border-slate-800"
                      >
                        <div className="text-lg font-black text-slate-100">{v}</div>
                        <p className="text-[9px] text-slate-600 uppercase font-black tracking-tighter text-center break-words">
                          {k}
                        </p>
                      </div>
                    ))}
                  </div>
                </div>

                {/* Latency Chart (원본의 별도 섹션 톤 반영) */}
                <section className="bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                  <h4 className="text-slate-500 text-[10px] font-black mb-6 uppercase text-center font-mono">
                    Physical Performance Comparison (Latency ms)
                  </h4>

                  <div className="h-52 sm:h-56">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={latencyData} layout="vertical">
                        <XAxis type="number" hide />
                        <YAxis dataKey="name" type="category" stroke="#94a3b8" fontSize={10} width={110} />
                        <Tooltip
                          cursor={{ fill: '#2d3748' }}
                          contentStyle={{
                            backgroundColor: '#0f172a',
                            border: '1px solid #334155',
                            borderRadius: '12px',
                          }}
                        />
                        <Bar dataKey="value" barSize={24} radius={[0, 6, 6, 0]}>
                          {latencyData.map((e, i) => <Cell key={i} fill={e.color} />)}
                        </Bar>
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </section>
              </div>
            </div>
          </section>
        </div>
      </main>

      <KernelDeepDive isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} data={data} />

      {/* ✅ 원본에 있던 scrollbar-hide 유지 */}
      <style jsx="true">{`
        .scrollbar-hide::-webkit-scrollbar { display: none; }
        .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
      `}</style>
    </div>
  );
}

function MetricCard({ title, value, color, icon }) {
  return (
    <div className="bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-lg transition hover:border-emerald-500/30">
      <div className="flex items-center gap-2 text-slate-500 mb-3 text-[10px] font-black uppercase tracking-widest">
        {icon} {title}
      </div>
      <p className={`text-3xl sm:text-4xl font-black font-mono tracking-tighter ${color}`}>
        {value ?? '—'}
      </p>
    </div>
  );
}