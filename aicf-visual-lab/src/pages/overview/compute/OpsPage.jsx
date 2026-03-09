// src/pages/OpsPage.jsx
import React, { useMemo, useState, useEffect } from 'react';
import 'katex/dist/katex.min.css';
import { InlineMath, BlockMath } from 'react-katex';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import {
  Cpu,
  Zap,
  Share2,
  Layers,
  ShieldCheck,
  Activity,
  Terminal,
  Scale,
  Eye,
  Focus,
  History,
  Boxes,
  Menu,
  ArrowUpRight,
  GitMerge,
  Search,
} from 'lucide-react';

import { useSearchParams, Link } from 'react-router-dom';
import { allOpsData } from '../../../data/index.js';
import KernelDeepDive from '../../../components/KernelDeepDive.jsx';
import ComputeSidebar from '../../../components/ComputeSidebar.jsx';

export default function OpsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeOpId = searchParams.get('op');

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const isMain = !activeOpId;
  const data = activeOpId ? allOpsData[activeOpId] : null;

  // ✅ op 바뀌면 모바일 드로어 닫기
  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activeOpId]);

  // Error state for missing data
  if (activeOpId && !data) {
    return (
      <div className="p-10 text-blue-400 bg-[#0f172a] min-h-screen flex flex-col items-center justify-center font-mono">
        <div className="animate-pulse mb-4 text-2xl font-black uppercase">Analyzing...</div>
        <div className="text-slate-500 text-sm">해당 연산의 명세가 data/index.js에 존재하지 않습니다.</div>
        <Link
          to="/ops"
          className="mt-6 px-4 py-2 rounded-xl bg-blue-600 text-white font-bold"
        >
          Back to Ops Guide
        </Link>
      </div>
    );
  }

  const semantic = data?.semantics ?? data?.semantic ?? null;
  const formula = data?.canonical?.formula ?? '';
  const shapes = data?.canonical?.shapes ?? {};
  const interpretation = data?.canonical?.interpretation ?? {};
  const latency = data?.performance?.latency ?? {};
  const km = data?.kernel?.metrics ?? {};
  const chosenVariant = data?.lowering?.chosen?.variant ?? 'Standard_Kernel';
  const hasDeepDive = !!(data?.kernel_evolution || data?.evolution);

  const latencyData = useMemo(() => {
    if (!data?.performance?.latency) return [];
    return [
      { name: 'PyTorch', value: latency.pytorch ?? 0, color: '#475569' },
      { name: 'torch.compile', value: latency.torch_compile ?? 0, color: '#64748b' },
      { name: 'AICF Optimized', value: latency.ours ?? 0, color: '#3b82f6' },
    ];
  }, [data, latency.pytorch, latency.torch_compile, latency.ours]);

  const handleSelectOp = (id) => {
    setSearchParams({ op: id }, { replace: true });
    setIsSidebarOpen(false);
  };

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      {/* ✅ GLOBAL SIDEBAR (통일) */}
      <ComputeSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeOpId={activeOpId || ''}
        quickOps={['AdamStep', 'LayerNorm', 'Softmax', 'GEMM']}
        version="v1.0.4 Stable"
      />

      {/* MAIN CONTENT */}
      <main className="flex-1 flex flex-col min-w-0 font-sans">
        {/* Mobile Header */}
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-5 py-4">
            <Link to="/" className="flex items-center gap-2">
              <div className="bg-blue-600 p-2 rounded-xl">
                <Cpu size={18} className="text-white" />
              </div>
              <div className="leading-none">
                <div className="font-black text-blue-400 tracking-tight">AICF LAB</div>
                <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                  v1.0.4 Stable
                </div>
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

        {/* ✅ 모바일 상단바 공간 확보 */}
        <div className="md:hidden h-[68px]" />

        {/* ✅ 메인만 스크롤 */}
        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-14 pb-32 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          {isMain ? (
            /* ============================================================
               CASE 1: OPS EXPLORER MAIN (HOW IT WORKS GUIDE)
               ============================================================ */
            <div className="max-w-5xl mx-auto space-y-20 animate-in fade-in duration-700">
              {/* Hero Section */}
              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-16 shadow-2xl relative overflow-hidden">
                <div className="absolute -top-10 -right-10 text-[140px] font-black text-blue-500/5 pointer-events-none">
                  OPS
                </div>
                <div className="flex items-center gap-2 text-blue-500 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
                  <Search size={16} /> Ops Explorer Guide
                </div>
                <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                  Graph Optimization & <br />
                  <span className="text-blue-500 text-3xl sm:text-5xl">Kernel Lowering</span>
                </h1>
                <p className="mt-8 text-slate-400 text-lg sm:text-xl leading-relaxed max-w-3xl font-light">
                  Ops Explorer는 연산이 <strong>&apos;어떻게(How)&apos;</strong> 실행되는지를 분석합니다.
                  <br />
                  단순한 구현을 넘어, 상위 그래프의 의미론(Theory)을 하드웨어의 물리적 성능으로 연결하는
                  최적화 경로를 추적합니다.
                </p>
              </section>

              {/* Core Philosophy: Theory vs Ops */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                  {
                    title: 'Semantic Anchor',
                    icon: ShieldCheck,
                    desc: 'Theory에서 정의된 수학적 불변성을 바탕으로 최적화의 정당성을 확보합니다.',
                  },
                  {
                    title: 'Chain Strategy',
                    icon: GitMerge,
                    desc: '인접한 연산 간의 데이터 흐름을 분석하여 커널 퓨전 및 메모리 재사용 계획을 수립합니다.',
                  },
                  {
                    title: 'Hardware Mapping',
                    icon: Cpu,
                    desc: 'GPU 아키텍처 특성에 맞춰 가장 효율적인 로우레벨 커널 변동사항(Variant)을 선택합니다.',
                  },
                ].map((item, idx) => (
                  <div
                    key={idx}
                    className="bg-[#0b1120] border border-slate-800 p-8 rounded-3xl hover:border-blue-500/30 transition"
                  >
                    <item.icon className="text-blue-500 mb-6" size={28} />
                    <h3 className="text-lg font-black text-white uppercase mb-3">{item.title}</h3>
                    <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
                  </div>
                ))}
              </div>

              {/* Structure of Analysis */}
              <section className="space-y-8">
                <div className="flex items-center gap-3 text-emerald-400">
                  <Terminal size={24} />
                  <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                    Analysis Framework
                  </h2>
                </div>

                <div className="space-y-4">
                  {[
                    {
                      step: '1',
                      title: '연산 본질 정의',
                      desc: '입출력 텐서의 Shape과 수식을 통해 연산의 입출력 관계를 규명합니다.',
                      tag: 'INPUT/OUTPUT',
                    },
                    {
                      step: '2',
                      title: '연쇄 최적화 전략',
                      desc: 'Downstream 연산에 대한 민감도를 분석하여 최적의 Lowering 전략을 결정합니다.',
                      tag: 'OPTIMIZATION',
                    },
                    {
                      step: '3',
                      title: '하드웨어 매핑',
                      desc: '실제 하드웨어에서의 처리량(Throughput)과 메모리 재사용률을 측정합니다.',
                      tag: 'HARDWARE',
                    },
                    {
                      step: '4',
                      title: '성능 비교',
                      desc: 'Baseline(PyTorch) 대비 최적화된 결과의 지연시간(Latency)을 시각화합니다.',
                      tag: 'BENCHMARK',
                    },
                  ].map((item) => (
                    <div
                      key={item.step}
                      className="flex flex-col sm:flex-row gap-6 bg-[#1e293b] p-8 rounded-[2rem] border border-slate-800 items-start sm:items-center"
                    >
                      <div className="w-12 h-12 bg-blue-600 rounded-2xl flex items-center justify-center font-black text-xl shrink-0 shadow-lg shadow-blue-600/20">
                        {item.step}
                      </div>
                      <div className="flex-1 min-w-0">
                        <div className="text-[10px] font-black text-blue-500 uppercase tracking-widest mb-1">
                          {item.tag}
                        </div>
                        <h4 className="text-xl font-black text-white mb-1 uppercase tracking-tight">
                          {item.title}
                        </h4>
                        <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
                      </div>
                      <div className="hidden lg:block">
                        <ArrowUpRight className="text-slate-700" size={24} />
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* CTA Section */}
              <section className="bg-blue-600/5 border border-blue-500/20 rounded-[3rem] p-12 text-center">
                <h2 className="text-2xl font-black text-white uppercase mb-8">
                  Select an Operator to Explore
                </h2>
                <div className="flex flex-wrap justify-center gap-4">
                  {Object.keys(allOpsData)
                    .slice(0, 6)
                    .map((id) => (
                      <button
                        key={id}
                        onClick={() => handleSelectOp(id)}
                        className="px-6 py-3 bg-[#0f172a] border border-slate-700 rounded-2xl text-blue-300 font-bold hover:border-blue-500 transition shadow-xl uppercase tracking-wider text-xs"
                      >
                        {id}
                      </button>
                    ))}
                </div>

                <div className="mt-6 text-[11px] text-slate-500">
                  더 많은 연산은 좌측 사이드바(모바일: 우측 상단 메뉴)에서 선택하세요.
                </div>
              </section>
            </div>
          ) : (
            /* ============================================================
               CASE 2: OPS DETAIL ANALYSIS (FULL CONTENT)
               ============================================================ */
            <div className="animate-in slide-in-from-bottom-4 duration-500 space-y-12">
              {/* Header Title Section */}
              <section className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 border-b border-slate-800 pb-8">
                <div className="space-y-2 min-w-0">
                  <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black uppercase tracking-[0.3em]">
                    <Activity size={14} /> Architecture Trace Report
                  </div>
                  <h2 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight break-words">
                    {data.id}{' '}
                    <span className="text-blue-500/30 font-light ml-2">Explorer</span>
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
                  {data.descriptions?.essence ?? '해당 연산의 수학적/의미론적 본질을 분석합니다.'}
                </p>

                <div className="grid grid-cols-12 gap-6">
                  <div className="col-span-12 lg:col-span-8 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                    {/* Formula */}
                    <div className="bg-[#0b1120] p-6 sm:p-10 rounded-3xl border border-slate-800/50 mb-8 overflow-x-auto scrollbar-hide">
                      <div className="text-3xl sm:text-4xl text-blue-400 text-center min-w-max">
                        <BlockMath math={formula} />
                      </div>

                      {/* Shapes */}
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
                            {interpretation?.[axis] || '정의되지 않음'}
                          </div>

                          <div className="relative z-10 mt-4 pt-3 border-t border-slate-800/60 text-[10px] text-slate-500">
                            <span className="text-slate-400 font-black uppercase mr-1">
                              Role:
                            </span>
                            &quot;{semantic.axes[axis].role}&quot;
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
                  <h3 className="text-2xl font-black uppercase tracking-tight">2. 연쇄 최적화 전략</h3>
                </div>

                <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
                  {data.descriptions?.strategy ??
                    '인접한 후행 연산의 특성을 분석하여 최적화 경로를 탐색합니다.'}
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
                  {data.descriptions?.hardware ?? 'GPU 하드웨어 매핑 전략을 수행합니다.'}
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  <div className="lg:col-span-5 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                    <div className="flex items-center gap-2 mb-6 text-emerald-400 font-black text-[10px] uppercase tracking-widest">
                      <Terminal size={16} /> Lowering Decision Engine
                    </div>

                    <p className="text-[10px] text-slate-500 uppercase font-black mb-2">
                      Selected Variant
                    </p>
                    <div className="text-2xl font-black text-white mb-6 break-words">
                      &quot;{chosenVariant}&quot;
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
                        {Object.entries(data.costModel?.weights_hint?.default ?? {}).map(
                          ([k, v]) => (
                            <div
                              key={k}
                              className="flex flex-col items-center gap-2 p-4 bg-[#0f172a]/50 rounded-2xl border border-slate-800"
                            >
                              <div className="text-lg font-black text-slate-100">{v}</div>
                              <p className="text-[9px] text-slate-600 uppercase font-black tracking-tighter text-center break-words">
                                {k}
                              </p>
                            </div>
                          )
                        )}
                      </div>
                    </div>

                    <section className="bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                      <h4 className="text-slate-500 text-[10px] font-black mb-6 uppercase text-center font-mono">
                        Physical Performance Comparison (Latency ms)
                      </h4>

                      <div className="h-52 sm:h-56">
                        <ResponsiveContainer width="100%" height="100%">
                          <BarChart data={latencyData} layout="vertical">
                            <XAxis type="number" hide />
                            <YAxis
                              dataKey="name"
                              type="category"
                              stroke="#94a3b8"
                              fontSize={10}
                              width={110}
                            />
                            <Tooltip
                              cursor={{ fill: '#2d3748' }}
                              contentStyle={{
                                backgroundColor: '#0f172a',
                                border: '1px solid #334155',
                                borderRadius: '12px',
                              }}
                            />
                            <Bar dataKey="value" barSize={24} radius={[0, 6, 6, 0]}>
                              {latencyData.map((e, i) => (
                                <Cell key={i} fill={e.color} />
                              ))}
                            </Bar>
                          </BarChart>
                        </ResponsiveContainer>
                      </div>
                    </section>
                  </div>
                </div>
              </section>

              {/* Footer: Back + Theory */}
              <div className="pt-10 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
                <Link
                  to="/ops"
                  className="text-slate-500 font-bold hover:text-white transition uppercase text-xs tracking-widest flex items-center gap-2"
                  onClick={(e) => {
                    e.preventDefault();
                    setSearchParams({}, { replace: true });
                  }}
                >
                  ← Back to Explorer Guide
                </Link>

                <Link
                  to={`/theory?op=${data.id}`}
                  className="px-6 py-3 rounded-2xl bg-blue-600/10 border border-blue-500/20 text-blue-300 font-black text-xs uppercase tracking-widest hover:bg-blue-600/20 transition flex items-center gap-2"
                >
                  View Theory Spec <ArrowUpRight size={16} />
                </Link>
              </div>
            </div>
          )}
        </div>
      </main>

      {/* Deep Dive Modal */}
      <KernelDeepDive isOpen={isModalOpen} onClose={() => setIsModalOpen(false)} data={data} />

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