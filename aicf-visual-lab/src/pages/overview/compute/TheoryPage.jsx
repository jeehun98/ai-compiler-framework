// src/pages/overview/compute/TheoryPage.jsx
import React, { useEffect, useMemo, useState } from "react";
import "katex/dist/katex.min.css";
import { BlockMath, InlineMath } from "react-katex";
import { Link, useSearchParams } from "react-router-dom";
import {
  Cpu,
  Menu,
  ShieldCheck,
  Target,
  Binary,
  ArrowRightLeft,
  Waypoints,
  Shrink,
  BookOpen,
  XCircle,
  Layers,
  Info,
  ArrowRight,
  Scale,
  Zap,
  Boxes,
  GitMerge,
  Workflow,
} from "lucide-react";

import ComputeSidebar from "../../../components/layout/ComputeSidebar.jsx";
import {
  theoryByPropertyId,
  theoryPropertyIds,
} from "../../../data/theory/index.js";

// Property card icons
const iconMap = {
  target: Target,
  binary: Binary,
  arrow: ArrowRightLeft,
  shield: ShieldCheck,
  merge: GitMerge,
  boxes: Boxes,
  workflow: Workflow,
  zap: Zap,
};

function IconBadge({ icon }) {
  const Icon = iconMap[icon] ?? ShieldCheck;
  return (
    <div className="p-3 bg-[#0f172a] rounded-2xl border border-slate-800">
      <Icon size={18} className="text-blue-400" />
    </div>
  );
}

export default function TheoryPage() {
  const [searchParams] = useSearchParams();
  const activePropertyId = searchParams.get("property");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const quickProperties = useMemo(
    () => [
      "OrderRewritable",
      "TileComposable",
      "LayoutFlexible",
      "DomainPrunable",
    ],
    []
  );

  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activePropertyId]);

  const isMain = !activePropertyId;
  const spec = activePropertyId ? theoryByPropertyId[activePropertyId] : null;

  if (activePropertyId && !spec) {
    return (
      <div className="p-10 text-blue-400 bg-[#0f172a] min-h-screen flex flex-col items-center justify-center font-mono">
        <div className="animate-pulse mb-4 text-2xl font-black uppercase">
          Property Not Found
        </div>
        <div className="text-slate-500 text-sm">
          data/theory/index.js에 "{activePropertyId}" 스펙이 없습니다.
        </div>
        <Link
          to="/compute/theory"
          className="mt-6 px-4 py-2 rounded-xl bg-blue-600 text-white font-bold"
        >
          Back to Property Guide
        </Link>
      </div>
    );
  }

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
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
                v1.1.0 Property View
              </div>
            </div>
          </Link>

          <button
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 rounded-xl border border-slate-700 bg-[#1e293b]"
            aria-label="Open sidebar"
          >
            <Menu size={18} />
          </button>
        </div>
      </header>

      <ComputeSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.1.0 Property View"
      />

      <main className="flex-1 flex flex-col min-w-0">
        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-14 pb-32 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          {isMain ? (
            <div className="max-w-6xl mx-auto space-y-20 animate-in fade-in duration-700">
              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-16 shadow-2xl relative overflow-hidden">
                <div className="absolute -top-10 -right-10 text-[140px] font-black text-blue-500/5 pointer-events-none">
                  THEORY
                </div>

                <div className="flex items-center gap-2 text-blue-500 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
                  <BookOpen size={16} /> Compute Property Atlas
                </div>

                <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                  Mathematical Semantics <br />
                  <span className="text-blue-500 text-3xl sm:text-5xl">
                    of Compute Properties
                  </span>
                </h1>

                <p className="mt-8 text-slate-400 text-lg sm:text-xl leading-relaxed max-w-4xl">
                  Theory는 이제 개별 operator를 직접 설명하는 페이지가 아니라,{" "}
                  <strong>
                    runtime transformation을 가능하게 하는 semantic property
                  </strong>
                  를 정의하는 계층입니다.
                  <br />
                  이 페이지는 어떤 재배치, 분해, 타일링, 특수화가 의미 보존
                  하에서 허용되는지의 수학적 조건을 다룹니다.
                </p>

                <div className="mt-8 inline-flex items-center gap-2 rounded-2xl border border-blue-500/20 bg-blue-500/5 px-4 py-2 text-[11px] font-bold uppercase tracking-widest text-blue-300">
                  <ShieldCheck size={14} />
                  Semantic Conditions for Runtime Transformation
                </div>
              </section>

              <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-[#0b1120] border border-blue-500/20 rounded-[2rem] p-8">
                  <div className="flex items-center gap-3 text-blue-400 mb-6">
                    <Target size={24} />
                    <h2 className="text-xl font-black uppercase">
                      What Theory Covers Now
                    </h2>
                  </div>

                  <ul className="space-y-4 text-slate-300">
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-blue-500 rounded-full shrink-0" />
                      <span>
                        <strong>Property Definition:</strong> 어떤 불변 성질이
                        연산 변환을 가능하게 하는지 정의합니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-blue-500 rounded-full shrink-0" />
                      <span>
                        <strong>Legality Condition:</strong> 어떤 조건 아래에서
                        transform이 의미 보존적으로 성립하는지 다룹니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-blue-500 rounded-full shrink-0" />
                      <span>
                        <strong>Enabled Transform Family:</strong> tiling,
                        pruning, specialization 같은 허용 transform을
                        연결합니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-blue-500 rounded-full shrink-0" />
                      <span>
                        <strong>Representative Operators:</strong> 각 property를
                        대표하는 operator / subgraph 예시를 제공합니다.
                      </span>
                    </li>
                  </ul>
                </div>

                <div className="bg-[#0b1120] border border-red-500/10 rounded-[2rem] p-8">
                  <div className="flex items-center gap-3 text-red-400 mb-6">
                    <XCircle size={24} />
                    <h2 className="text-xl font-black uppercase text-slate-300">
                      What It Does NOT Contain
                    </h2>
                  </div>

                  <ul className="space-y-4 text-slate-500 italic">
                    <li className="flex gap-3 line-through decoration-slate-700">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-red-900/50 rounded-full shrink-0" />
                      <span>개별 operator의 상세 구현 설명</span>
                    </li>
                    <li className="flex gap-3 line-through decoration-slate-700">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-red-900/50 rounded-full shrink-0" />
                      <span>커널 내부 scheduling, CTA/warp micro-optimization</span>
                    </li>
                    <li className="flex gap-3 line-through decoration-slate-700">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-red-900/50 rounded-full shrink-0" />
                      <span>벤치마크 수치, profiler metric, hardware report</span>
                    </li>
                  </ul>

                  <p className="mt-10 p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-[12px] text-slate-400 leading-relaxed font-medium">
                    <span className="text-blue-400 font-bold block mb-1">
                      NOTE: Relationship with Ops Explorer
                    </span>
                    Theory는 <strong>property</strong>를 정의하고, Ops Explorer는
                    각 operator가 어떤 <strong>property profile</strong>을
                    갖는지와 그 결과 어떤 lowering family로 이어지는지를
                    보여줍니다.
                  </p>
                </div>
              </section>

              <section className="space-y-8">
                <div className="flex items-center gap-3 text-emerald-400">
                  <Layers size={24} />
                  <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                    Property Spec Structure
                  </h2>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
                  {[
                    {
                      id: "01",
                      title: "Definition",
                      desc: "property의 의미와 수학적 형태",
                      icon: Binary,
                    },
                    {
                      id: "02",
                      title: "Legality",
                      desc: "언제 transform이 허용되는가",
                      icon: ShieldCheck,
                    },
                    {
                      id: "03",
                      title: "Enables",
                      desc: "가능한 runtime / lowering family",
                      icon: Zap,
                    },
                    {
                      id: "04",
                      title: "Boundary",
                      desc: "깨지는 조건과 한계",
                      icon: Waypoints,
                    },
                    {
                      id: "05",
                      title: "Ops Mapping",
                      desc: "대표 operator / subgraph 예시",
                      icon: Boxes,
                    },
                  ].map((item) => (
                    <div
                      key={item.id}
                      className="bg-[#1e293b] p-8 rounded-[2rem] border border-slate-800 hover:border-blue-500/30 transition shadow-lg group"
                    >
                      <item.icon
                        className="text-blue-500 mb-6 group-hover:scale-110 transition-transform"
                        size={28}
                      />
                      <div className="text-[10px] font-mono text-slate-600 mb-1 font-bold">
                        SECTION {item.id}
                      </div>
                      <h3 className="font-black text-white uppercase text-lg mb-2">
                        {item.title}
                      </h3>
                      <p className="text-slate-400 text-sm leading-relaxed">
                        {item.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </section>

              <section className="space-y-8">
                <div className="flex items-center gap-3 text-blue-400">
                  <Workflow size={24} />
                  <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                    Compute Properties Covered
                  </h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-5">
                  {theoryPropertyIds.map((id) => {
                    const item = theoryByPropertyId[id];
                    return (
                      <Link
                        key={id}
                        to={`/compute/theory?property=${id}`}
                        className="group bg-[#1e293b] border border-slate-800 rounded-[2rem] p-6 hover:border-blue-500/30 transition shadow-xl"
                      >
                        <div className="text-[10px] font-black uppercase tracking-widest text-blue-500 mb-2">
                          {item.subtitle || "Compute Property"}
                        </div>

                        <div className="flex items-start justify-between gap-4">
                          <div className="min-w-0">
                            <h3 className="text-xl font-black text-white uppercase tracking-tight break-words">
                              {item.title}
                            </h3>
                            <p className="mt-3 text-sm text-slate-400 leading-relaxed">
                              {item.hero?.lead ||
                                "Semantic condition that enables invariant-preserving runtime transformation."}
                            </p>
                          </div>
                          <ArrowRight
                            size={18}
                            className="text-slate-600 group-hover:text-blue-400 transition shrink-0"
                          />
                        </div>
                      </Link>
                    );
                  })}
                </div>
              </section>

              <section className="bg-blue-600/5 border border-blue-500/20 rounded-[3rem] p-12">
                <div className="max-w-4xl">
                  <div className="text-[11px] font-black uppercase tracking-widest text-blue-400 mb-4">
                    Core Principle
                  </div>

                  <h2 className="text-2xl sm:text-3xl font-black text-white uppercase mb-5 leading-tight">
                    Property is the reason. Transform is the action.
                  </h2>

                  <p className="text-slate-400 leading-relaxed text-base sm:text-lg">
                    AICF에서 optimization은 operator 이름으로 정의되지 않고,
                    연산이 가진 semantic property 위에서 정의됩니다. 즉,
                    property는 <strong>왜 바꿔도 되는가</strong>를 설명하고,
                    transform은 <strong>무엇을 바꾸는가</strong>를 설명합니다.
                  </p>
                </div>
              </section>

              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 text-center">
                <div className="flex items-center justify-center gap-2 text-emerald-400 mb-4">
                  <ArrowRight size={18} />
                  <span className="text-[11px] font-black uppercase tracking-widest">
                    Next Layer
                  </span>
                </div>

                <h2 className="text-2xl font-black text-white uppercase mb-4">
                  See How Properties Appear in Real Operators
                </h2>

                <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed mb-8">
                  Theory가 property를 정의했다면, Ops Explorer는 각 operator가
                  어떤 property 조합을 가지며 어떤 lowering family로 이어지는지
                  보여줍니다.
                </p>

                <Link
                  to="/compute/ops"
                  className="inline-flex items-center gap-2 px-8 py-4 rounded-2xl bg-blue-600/10 border border-blue-500/20 text-blue-300 font-black uppercase tracking-widest hover:bg-blue-600/20 transition"
                >
                  Go to Ops Explorer <ArrowRight size={16} />
                </Link>
              </section>
            </div>
          ) : (
            <div className="space-y-14 animate-in slide-in-from-bottom-4 duration-500">
              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-12 shadow-2xl relative overflow-hidden">
                <div className="absolute -top-10 -right-10 text-[120px] sm:text-[160px] font-black text-blue-500/5 pointer-events-none tracking-tighter uppercase">
                  {spec.id}
                </div>

                <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black uppercase tracking-[0.35em]">
                  <Waypoints size={14} /> {spec.subtitle || "Compute Property"}
                </div>

                <h1 className="mt-4 text-4xl sm:text-6xl font-black tracking-tight text-white leading-[1.05]">
                  {spec.title}
                </h1>

                {spec.hero?.lead && (
                  <p className="mt-6 max-w-4xl text-slate-400 text-base sm:text-lg leading-relaxed italic">
                    {spec.hero.lead}
                  </p>
                )}

                {spec.hero?.canonicalLatex && (
                  <div className="mt-8 bg-[#0b1120] border border-slate-800/50 rounded-3xl p-6 sm:p-8 overflow-x-auto scrollbar-hide shadow-inner">
                    <div className="w-max min-w-full text-blue-300 text-center text-2xl sm:text-3xl">
                      <BlockMath math={spec.hero.canonicalLatex} />
                    </div>
                  </div>
                )}
              </section>

              {spec.sections?.definition && (
                <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                  <div className="lg:col-span-5 space-y-6 min-w-0">
                    <div className="flex items-center gap-3 text-purple-400">
                      <Shrink size={22} />
                      <h2 className="text-2xl font-black uppercase tracking-tight">
                        Definition
                      </h2>
                    </div>

                    <div className="space-y-3 text-slate-400 leading-relaxed">
                      {spec.sections.definition.bullets?.map((b, i) => (
                        <div key={i} className="flex gap-3">
                          <div className="mt-2 w-1.5 h-1.5 bg-purple-500 rounded-full shrink-0" />
                          <div>
                            <span className="text-slate-200 font-bold">
                              {b.k}
                            </span>{" "}
                            — {b.v}
                          </div>
                        </div>
                      ))}
                    </div>

                    {spec.sections.definition.latex && (
                      <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 shadow-xl overflow-x-auto scrollbar-hide min-w-0">
                        <div className="w-max min-w-full">
                          <BlockMath math={spec.sections.definition.latex} />
                        </div>
                        <p className="mt-4 text-[11px] text-center text-slate-500 uppercase font-bold tracking-widest">
                          Canonical Property Form
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="lg:col-span-7 bg-[#0b1120] rounded-[2.5rem] border border-blue-500/20 p-8 min-w-0">
                    <div className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-4 flex items-center gap-2">
                      <Info size={12} /> Why It Matters
                    </div>

                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {spec.sections.definition.preview?.map((r, i) => (
                        <div
                          key={i}
                          className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 hover:border-blue-500/30 transition"
                        >
                          <div className="text-slate-200 font-black uppercase tracking-tight">
                            {r.k}
                          </div>
                          <p className="mt-2 text-sm text-slate-400 leading-relaxed">
                            {r.v}
                          </p>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              )}

              {spec.sections?.legality && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-emerald-400">
                    <ShieldCheck size={22} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                      Legality Conditions
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {spec.sections.legality.cards?.map((c) => (
                      <div
                        key={c.id}
                        className="bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-xl group hover:border-emerald-500/30 transition min-w-0"
                      >
                        <div className="flex items-center justify-between mb-6">
                          <IconBadge icon={c.icon} />
                          <span className="text-[10px] font-mono text-slate-600 font-bold tracking-widest uppercase">
                            Rule {c.id}
                          </span>
                        </div>

                        <h3 className="text-xl font-black text-white mb-3 italic uppercase tracking-tighter">
                          {c.title}
                        </h3>

                        <p className="text-slate-400 text-sm leading-relaxed mb-5">
                          {c.desc}
                        </p>

                        {c.metric && (
                          <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 text-[11px] text-blue-200/70 font-mono italic min-w-0">
                            <div className="overflow-x-auto scrollbar-hide">
                              <div className="w-max">
                                <InlineMath math={c.metric} />
                              </div>
                            </div>
                          </div>
                        )}

                        {c.note && (
                          <div className="mt-4 text-[11px] text-emerald-400 font-bold uppercase tracking-widest">
                            {c.note}
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {spec.sections?.enables && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-blue-400">
                    <Zap size={22} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                      Enabled Transform Families
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                    {spec.sections.enables.items?.map((item, i) => (
                      <div
                        key={i}
                        className="bg-[#1e293b] border border-slate-800 rounded-[2rem] p-6"
                      >
                        <div className="text-[10px] font-black uppercase tracking-widest text-blue-500 mb-2">
                          Transform {String(i + 1).padStart(2, "0")}
                        </div>
                        <div className="text-lg font-black text-white uppercase tracking-tight">
                          {item}
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {spec.sections?.boundary && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-amber-400">
                    <Scale size={22} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                      Boundary Conditions
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
                    {spec.sections.boundary.items?.map((item, i) => (
                      <div
                        key={i}
                        className="bg-[#0f172a] border border-slate-800 rounded-2xl p-6"
                      >
                        <div className="text-[10px] font-black uppercase tracking-widest text-amber-400 mb-2">
                          Boundary {String(i + 1).padStart(2, "0")}
                        </div>
                        <p className="text-sm text-slate-400 leading-relaxed">
                          {item}
                        </p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              <section className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                {spec.sections?.relatedOps && (
                  <div className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-8">
                    <div className="flex items-center gap-3 text-purple-400 mb-6">
                      <Boxes size={22} />
                      <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                        Representative Ops
                      </h2>
                    </div>

                    <div className="flex flex-wrap gap-3">
                      {spec.sections.relatedOps.items?.map((op) => (
                        <Link
                          key={op}
                          to={`/compute/ops?op=${op}`}
                          className="px-4 py-2 rounded-xl bg-[#0f172a] border border-slate-700 text-blue-300 font-black uppercase tracking-wider text-xs hover:border-blue-500 transition"
                        >
                          {op}
                        </Link>
                      ))}
                    </div>
                  </div>
                )}

                {spec.sections?.relatedTransforms && (
                  <div className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-8">
                    <div className="flex items-center gap-3 text-emerald-400 mb-6">
                      <GitMerge size={22} />
                      <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                        Runtime / Lowering Links
                      </h2>
                    </div>

                    <div className="space-y-3">
                      {spec.sections.relatedTransforms.items?.map((t, i) => (
                        <div
                          key={i}
                          className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 text-sm text-slate-300 font-bold"
                        >
                          {t}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>

              <div className="pt-10 border-t border-slate-800 flex flex-col sm:flex-row justify-center items-center gap-4">
                <Link
                  to="/compute/theory"
                  className="flex items-center gap-2 text-blue-400 font-black uppercase text-sm hover:text-white transition"
                >
                  <BookOpen size={16} /> Back to Property Guide
                </Link>

                <Link
                  to="/compute/ops"
                  className="flex items-center gap-2 text-emerald-400 font-black uppercase text-sm hover:text-white transition"
                >
                  <ArrowRight size={16} /> View Ops Explorer
                </Link>
              </div>
            </div>
          )}
        </div>
      </main>

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