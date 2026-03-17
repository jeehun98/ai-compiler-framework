// src/pages/OpsPage.jsx
import React, { useMemo, useState, useEffect } from "react";
import "katex/dist/katex.min.css";
import { InlineMath, BlockMath } from "react-katex";
import {
  Cpu,
  Zap,
  Share2,
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
  BookOpen,
  ArrowRightLeft,
} from "lucide-react";

import { useSearchParams, Link } from "react-router-dom";
import { allOpsData } from "../../../data/ops/index.js";
import KernelDeepDive from "../../../components/common/KernelDeepDive.jsx";
import ComputeSidebar from "../../../components/layout/ComputeSidebar.jsx";

export default function OpsPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const activeOpId = searchParams.get("op");

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const isMain = !activeOpId;
  const data = activeOpId ? allOpsData[activeOpId] : null;

  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activeOpId]);

  if (activeOpId && !data) {
    return (
      <div className="p-10 text-blue-400 bg-[#0f172a] min-h-screen flex flex-col items-center justify-center font-mono">
        <div className="animate-pulse mb-4 text-2xl font-black uppercase">
          Analyzing...
        </div>
        <div className="text-slate-500 text-sm">
          해당 연산의 명세가 data/index.js에 존재하지 않습니다.
        </div>
        <Link
          to="/compute/ops"
          className="mt-6 px-4 py-2 rounded-xl bg-blue-600 text-white font-bold"
        >
          Back to Ops Guide
        </Link>
      </div>
    );
  }

  const semantic = data?.semantics ?? data?.semantic ?? null;
  const formula = data?.canonical?.formula ?? "";
  const shapes = data?.canonical?.shapes ?? {};
  const interpretation = data?.canonical?.interpretation ?? {};
  const chosenVariant = data?.lowering?.chosen?.variant ?? "Standard_Kernel";
  const hasDeepDive = !!(data?.kernel_evolution || data?.evolution);

  const invariants = semantic?.invariants ?? [];
  const downstream = semantic?.sensitivity?.downstream ?? [];
  const loweringReasons = data?.lowering?.chosen?.reason ?? [];
  const costWeights = data?.costModel?.weights_hint?.default ?? {};

  const quickOps = useMemo(
    () => ["AdamStep", "LayerNorm", "Softmax", "GEMM"],
    []
  );

  const handleSelectOp = (id) => {
    setSearchParams({ op: id }, { replace: true });
    setIsSidebarOpen(false);
  };

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <ComputeSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeOpId={activeOpId || ""}
        quickOps={quickOps}
        version="v1.0.5 Semantic View"
      />

      <main className="flex-1 flex flex-col min-w-0 font-sans">
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-5 py-4">
            <Link to="/" className="flex items-center gap-2">
              <div className="bg-blue-600 p-2 rounded-xl">
                <Cpu size={18} className="text-white" />
              </div>
              <div className="leading-none">
                <div className="font-black text-blue-400 tracking-tight">
                  AICF LAB
                </div>
                <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                  v1.0.5 Semantic View
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

        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-14 pb-32 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          {isMain ? (
            <div className="max-w-5xl mx-auto space-y-20 animate-in fade-in duration-700">
              {/* Hero */}
              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-16 shadow-2xl relative overflow-hidden">
                <div className="absolute -top-10 -right-10 text-[140px] font-black text-blue-500/5 pointer-events-none">
                  OPS
                </div>

                <div className="flex items-center gap-2 text-blue-500 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
                  <Search size={16} /> Ops Explorer Guide
                </div>

                <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                  Semantic Meaning & <br />
                  <span className="text-blue-500 text-3xl sm:text-5xl">
                    Lowering Outlook
                  </span>
                </h1>

                <p className="mt-8 text-slate-400 text-lg sm:text-xl leading-relaxed max-w-3xl font-light">
                  Ops Explorer는 각 연산이 가지는{" "}
                  <strong>수학적 의미와 불변성</strong>을 기준으로, 어떤
                  최적화가 허용되며 어떤 실행 형태로 이어질 수 있는지를
                  분석합니다.
                  <br />
                  즉, Theory가 정의한 의미를 실제 optimization candidate로
                  연결하는{" "}
                  <strong>semantic-to-lowering bridge</strong> 역할을 합니다.
                </p>

                <div className="mt-8 inline-flex items-center gap-2 rounded-2xl border border-blue-500/20 bg-blue-500/5 px-4 py-2 text-[11px] font-bold uppercase tracking-widest text-blue-300">
                  <ShieldCheck size={14} />
                  Semantic / Compute View
                </div>
              </section>

              {/* Bridge */}
              <section className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 sm:p-10">
                <div className="flex items-center gap-3 text-emerald-400 mb-6">
                  <GitMerge size={22} />
                  <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                    From Theory to Realization
                  </h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    {
                      step: "01",
                      title: "Theory Spec",
                      desc: "연산의 수학적 정의와 보존 성질을 규정합니다.",
                    },
                    {
                      step: "02",
                      title: "Invariant Space",
                      desc: "어떤 재배치, 융합, 근사가 의미적으로 허용되는지 판별합니다.",
                    },
                    {
                      step: "03",
                      title: "Lowering Family",
                      desc: "허용된 공간 안에서 어떤 realization family가 자연스러운지 좁혀갑니다.",
                    },
                  ].map((item) => (
                    <div
                      key={item.step}
                      className="bg-[#111827] border border-slate-800 rounded-2xl p-5"
                    >
                      <div className="text-[10px] font-black uppercase tracking-widest text-blue-500 mb-2">
                        Step {item.step}
                      </div>
                      <h3 className="text-white font-black uppercase text-sm mb-2">
                        {item.title}
                      </h3>
                      <p className="text-slate-400 text-sm leading-relaxed">
                        {item.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </section>

              {/* Core cards */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {[
                  {
                    title: "Semantic Anchor",
                    icon: ShieldCheck,
                    desc: "연산이 반드시 보존해야 할 수학적 의미와 축별 역할을 정의합니다.",
                  },
                  {
                    title: "Invariant Space",
                    icon: GitMerge,
                    desc: "허용 가능한 최적화가 어떤 불변성 아래 성립하는지 분석합니다.",
                  },
                  {
                    title: "Lowering Outlook",
                    icon: Cpu,
                    desc: "의미 보존 하에서 가능한 realization family를 요약하고 다음 계층으로 연결합니다.",
                  },
                ].map((item, idx) => (
                  <div
                    key={idx}
                    className="bg-[#0b1120] border border-slate-800 p-8 rounded-3xl hover:border-blue-500/30 transition"
                  >
                    <item.icon className="text-blue-500 mb-6" size={28} />
                    <h3 className="text-lg font-black text-white uppercase mb-3">
                      {item.title}
                    </h3>
                    <p className="text-slate-400 text-sm leading-relaxed">
                      {item.desc}
                    </p>
                  </div>
                ))}
              </div>

              {/* Analysis Framework */}
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
                      step: "1",
                      title: "연산 의미 정의",
                      desc: "수식, 축 의미, 입출력 관계를 통해 연산이 보존해야 할 본질을 정의합니다.",
                      tag: "SEMANTICS",
                    },
                    {
                      step: "2",
                      title: "불변성 기반 제약 확인",
                      desc: "허용 가능한 최적화가 어떤 의미 보존 조건 아래 성립하는지 식별합니다.",
                      tag: "INVARIANTS",
                    },
                    {
                      step: "3",
                      title: "후행 연산 기반 Lowering 후보 탐색",
                      desc: "연쇄 구조와 downstream 민감도를 바탕으로 가능한 realization family를 좁힙니다.",
                      tag: "LOWERING",
                    },
                    {
                      step: "4",
                      title: "실행 형태 요약",
                      desc: "선택된 lowering family가 어떤 실행 스타일로 이어지는지 요약합니다. 세부 구현은 Deep Dive에서 다룹니다.",
                      tag: "REALIZATION",
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
                        <p className="text-slate-400 text-sm leading-relaxed">
                          {item.desc}
                        </p>
                      </div>
                      <div className="hidden lg:block">
                        <ArrowUpRight className="text-slate-700" size={24} />
                      </div>
                    </div>
                  ))}
                </div>
              </section>

              {/* Select operators */}
              <section className="bg-blue-600/5 border border-blue-500/20 rounded-[3rem] p-12 text-center">
                <h2 className="text-2xl font-black text-white uppercase mb-4">
                  Select an Operator to Explore
                </h2>
                <p className="text-slate-400 text-sm max-w-2xl mx-auto leading-relaxed mb-8">
                  각 연산이 어떤 의미를 가지며, 어떤 invariant를 통해 어떤
                  lowering family로 이어질 수 있는지 확인할 수 있습니다.
                </p>

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
                  더 많은 연산은 좌측 사이드바(모바일: 우측 상단 메뉴)에서
                  선택하세요.
                </div>
              </section>

              {/* CTA to Theory */}
              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 text-center">
                <div className="flex items-center justify-center gap-2 text-blue-400 mb-4">
                  <BookOpen size={18} />
                  <span className="text-[11px] font-black uppercase tracking-widest">
                    Semantic Basis
                  </span>
                </div>

                <h2 className="text-2xl font-black text-white uppercase mb-4">
                  Need the Mathematical Basis First?
                </h2>

                <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed mb-8">
                  Ops Explorer는 lowering과 optimization candidate를
                  설명합니다. 각 연산의 수학적 정의, 기하학적 해석, 등가
                  조건을 먼저 보려면 Theory Specs를 확인하세요.
                </p>

                <Link
                  to="/compute/theory"
                  className="inline-flex items-center gap-2 px-8 py-4 rounded-2xl bg-blue-600/10 border border-blue-500/20 text-blue-300 font-black uppercase tracking-widest hover:bg-blue-600/20 transition"
                >
                  View Theory Specs <ArrowUpRight size={16} />
                </Link>
              </section>
            </div>
          ) : (
            <div className="animate-in slide-in-from-bottom-4 duration-500 space-y-12">
              {/* Header */}
              <section className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 border-b border-slate-800 pb-8">
                <div className="space-y-2 min-w-0">
                  <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black uppercase tracking-[0.3em]">
                    <Activity size={14} /> Semantic Trace Report
                  </div>
                  <h2 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight break-words">
                    {data.id}{" "}
                    <span className="text-blue-500/30 font-light ml-2">
                      Explorer
                    </span>
                  </h2>
                </div>

                <div className="flex items-center gap-2 text-emerald-400 font-bold bg-emerald-400/5 px-4 py-2 rounded-xl border border-emerald-400/10 text-[11px] uppercase tracking-widest w-fit">
                  <ShieldCheck size={16} /> Semantic Anchored
                </div>
              </section>

              {/* Essence */}
              <section className="space-y-6">
                <div className="flex items-center gap-3 text-blue-400">
                  <Share2 size={24} />
                  <h3 className="text-2xl font-black uppercase tracking-tight">
                    1. 연산 본질 및 데이터 정의
                  </h3>
                </div>

                <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
                  {data.descriptions?.essence ??
                    "해당 연산의 수학적/의미론적 본질을 분석합니다."}
                </p>

                <div className="grid grid-cols-12 gap-6">
                  <div className="col-span-12 lg:col-span-8 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                    <div className="bg-[#0b1120] p-6 sm:p-10 rounded-3xl border border-slate-800/50 mb-8 overflow-x-auto scrollbar-hide">
                      <div className="text-3xl sm:text-4xl text-blue-400 text-center min-w-max">
                        <BlockMath math={formula} />
                      </div>

                      <div className="mt-6 flex flex-wrap justify-center gap-3 text-slate-500 font-mono text-xs">
                        {Object.entries(shapes).map(([tensor, shape]) => (
                          <div
                            key={tensor}
                            className="flex gap-2 items-center bg-[#0f172a] px-4 py-2 rounded-xl border border-slate-800 max-w-full"
                          >
                            <Boxes size={14} className="text-blue-500/40" />
                            <span className="text-blue-400 font-bold">
                              {tensor}:
                            </span>
                            <span className="italic break-all">{shape}</span>
                          </div>
                        ))}
                      </div>
                    </div>

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
                            {interpretation?.[axis] || "정의되지 않음"}
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

                  <div className="col-span-12 lg:col-span-4 space-y-4">
                    <div className="flex items-center justify-between px-2">
                      <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
                        Invariant-Preserving Optimization Space
                      </p>
                      <span className="text-[9px] text-emerald-500 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                        VERIFIED
                      </span>
                    </div>

                    {invariants.length > 0 ? (
                      invariants.map((inv) => (
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
                      ))
                    ) : (
                      <EmptyCard message="정의된 invariant가 없습니다." />
                    )}
                  </div>
                </div>
              </section>

              {/* Downstream */}
              <section className="space-y-6">
                <div className="flex items-center gap-3 text-purple-400">
                  <Eye size={24} />
                  <h3 className="text-2xl font-black uppercase tracking-tight">
                    2. Downstream-Aware Lowering Strategy
                  </h3>
                </div>

                <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
                  {data.descriptions?.strategy ??
                    "이 연산은 단독으로 최적화되지 않습니다. 후행 연산과의 결합 가능성, 축 민감도, 재사용 가능성을 함께 고려하여 lowering 후보가 결정됩니다."}
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {downstream.length > 0 ? (
                    downstream.map((ds, i) => (
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
                            <Zap size={14} />
                            <span className="break-words">{ds.hint}</span>
                          </div>
                        </div>
                      </div>
                    ))
                  ) : (
                    <EmptyCard
                      className="lg:col-span-2"
                      message="정의된 downstream lowering 규칙이 없습니다."
                    />
                  )}
                </div>
              </section>

              {/* Realization summary */}
              <section className="space-y-6">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-3">
                  <div className="flex items-center gap-3 text-emerald-400">
                    <Zap size={24} />
                    <h3 className="text-2xl font-black uppercase tracking-tight">
                      3. Execution Realization Summary
                    </h3>
                  </div>

                  {hasDeepDive && (
                    <button
                      onClick={() => setIsModalOpen(true)}
                      className="w-fit flex items-center gap-2 px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-white rounded-xl text-xs font-bold uppercase tracking-widest border border-slate-700 transition"
                    >
                      <History size={14} />
                      View Deep Dive
                    </button>
                  )}
                </div>

                <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
                  {data.descriptions?.hardware ??
                    "이 섹션은 연산 의미와 제약 조건으로부터 어떤 realization family가 선택되었는지를 요약합니다. 실제 메모리 스케줄, 커널 메트릭, 하드웨어 성능 수치는 별도의 Deep Dive에서 다룹니다."}
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  <div className="lg:col-span-6 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                    <div className="flex items-center gap-2 mb-6 text-emerald-400 font-black text-[10px] uppercase tracking-widest">
                      <Terminal size={16} /> Lowering Family
                    </div>

                    <p className="text-[10px] text-slate-500 uppercase font-black mb-2">
                      Selected Realization
                    </p>
                    <div className="text-2xl font-black text-white mb-6 break-words">
                      &quot;{chosenVariant}&quot;
                    </div>

                    <div className="space-y-3">
                      {loweringReasons.length > 0 ? (
                        loweringReasons.map((r, i) => (
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
                        ))
                      ) : (
                        <EmptyInner message="선택 이유가 정의되지 않았습니다." />
                      )}
                    </div>
                  </div>

                  <div className="lg:col-span-6 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                    <div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] font-black uppercase mb-6">
                      <Scale size={18} /> Semantic Cost Model
                    </div>

                    <div className="text-xs font-mono text-blue-400 italic break-words mb-6 min-h-[24px]">
                      {data.costModel?.semanticLoss ? (
                        <InlineMath math={data.costModel.semanticLoss} />
                      ) : (
                        <span className="text-slate-600 not-italic">
                          No semantic loss expression
                        </span>
                      )}
                    </div>

                    {Object.keys(costWeights).length > 0 ? (
                      <div className="grid grid-cols-3 gap-4">
                        {Object.entries(costWeights).map(([k, v]) => (
                          <div
                            key={k}
                            className="flex flex-col items-center gap-2 p-4 bg-[#0f172a]/50 rounded-2xl border border-slate-800"
                          >
                            <div className="text-lg font-black text-slate-100">
                              {v}
                            </div>
                            <p className="text-[9px] text-slate-600 uppercase font-black tracking-tighter text-center break-words">
                              {k}
                            </p>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <EmptyInner message="정의된 cost weight가 없습니다." />
                    )}

                    <div className="mt-6 pt-6 border-t border-slate-800">
                      <p className="text-xs text-slate-500 leading-relaxed">
                        Detailed memory scheduling, kernel metrics, and physical
                        performance results are handled in the Kernel / MCIR
                        Deep Dive layer.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              {/* Footer */}
              <div className="pt-10 border-t border-slate-800 flex flex-col sm:flex-row items-center justify-between gap-4">
                <Link
                  to="/compute/ops"
                  className="text-slate-500 font-bold hover:text-white transition uppercase text-xs tracking-widest flex items-center gap-2"
                  onClick={(e) => {
                    e.preventDefault();
                    setSearchParams({}, { replace: true });
                  }}
                >
                  ← Back to Explorer Guide
                </Link>

                <div className="flex flex-wrap items-center justify-center gap-3">
                  {hasDeepDive && (
                    <button
                      onClick={() => setIsModalOpen(true)}
                      className="px-6 py-3 rounded-2xl bg-slate-800 border border-slate-700 text-slate-200 font-black text-xs uppercase tracking-widest hover:bg-slate-700 transition flex items-center gap-2"
                    >
                      View Deep Dive <History size={16} />
                    </button>
                  )}

                  <Link
                    to={`/compute/theory?op=${data.id}`}
                    className="px-6 py-3 rounded-2xl bg-blue-600/10 border border-blue-500/20 text-blue-300 font-black text-xs uppercase tracking-widest hover:bg-blue-600/20 transition flex items-center gap-2"
                  >
                    View Theory Spec <ArrowUpRight size={16} />
                  </Link>
                </div>
              </div>
            </div>
          )}
        </div>
      </main>

      <KernelDeepDive
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        data={data}
      />

      <style jsx="true">{`
        .scrollbar-hide::-webkit-scrollbar {
          display: none;
        }
        .scrollbar-hide {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>
    </div>
  );
}

function EmptyCard({ message, className = "" }) {
  return (
    <div
      className={`bg-[#1e293b] p-8 rounded-[2.5rem] border border-dashed border-slate-700 text-slate-500 text-sm ${className}`}
    >
      {message}
    </div>
  );
}

function EmptyInner({ message }) {
  return (
    <div className="bg-[#0f172a] p-4 rounded-2xl border border-dashed border-slate-700 text-sm text-slate-500">
      {message}
    </div>
  );
}