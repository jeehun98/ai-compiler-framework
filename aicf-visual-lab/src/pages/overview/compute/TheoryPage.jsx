// src/pages/overview/compute/TheoryPage.jsx
import React, { useEffect, useState } from "react";
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
  theoryPropertyGroups,
} from "../../../data/theory/properties/index.js";

const quickProperties = [
  "OrderRewritable",
  "TileComposable",
  "RepresentationInvariant",
  "DomainPrunable",
];

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
    <div className="rounded-2xl border border-slate-800 bg-[#0f172a] p-3">
      <Icon size={18} className="text-blue-400" />
    </div>
  );
}

function ConstructionChip({ item }) {
  if (typeof item === "string") {
    return (
      <span className="rounded-xl border border-slate-700 bg-[#0f172a] px-4 py-2 text-xs font-black uppercase tracking-wider text-blue-300">
        {item}
      </span>
    );
  }

  if (item?.op) {
    return (
      <Link
        to={`/compute/ops?op=${item.op}`}
        className="rounded-xl border border-slate-700 bg-[#0f172a] px-4 py-2 text-xs font-black uppercase tracking-wider text-blue-300 transition hover:border-blue-500"
      >
        {item.label ?? item.op}
      </Link>
    );
  }

  return (
    <span className="rounded-xl border border-slate-700 bg-[#0f172a] px-4 py-2 text-xs font-black uppercase tracking-wider text-blue-300">
      {item?.label ?? "Unknown"}
    </span>
  );
}

function GroupBadge({ groupId }) {
  const meta =
    groupId === "foundational"
      ? {
          label: "Foundational",
          cls: "border-blue-500/20 bg-blue-500/5 text-blue-300",
        }
      : groupId === "reconstructive"
      ? {
          label: "Reconstructive",
          cls: "border-purple-500/20 bg-purple-500/5 text-purple-300",
        }
      : groupId === "structural"
      ? {
          label: "Structural",
          cls: "border-amber-500/20 bg-amber-500/5 text-amber-300",
        }
      : {
          label: "Property Group",
          cls: "border-slate-500/20 bg-slate-500/5 text-slate-300",
        };

  return (
    <span
      className={`inline-flex items-center rounded-xl border px-3 py-1 text-[10px] font-black uppercase tracking-widest ${meta.cls}`}
    >
      {meta.label}
    </span>
  );
}

function getGroupTheme(groupId) {
  if (groupId === "foundational") {
    return {
      headerText: "text-blue-400",
      cardHover: "hover:border-blue-500/30",
      arrowHover: "group-hover:text-blue-400",
    };
  }

  if (groupId === "reconstructive") {
    return {
      headerText: "text-purple-400",
      cardHover: "hover:border-purple-500/30",
      arrowHover: "group-hover:text-purple-400",
    };
  }

  if (groupId === "structural") {
    return {
      headerText: "text-amber-400",
      cardHover: "hover:border-amber-500/30",
      arrowHover: "group-hover:text-amber-400",
    };
  }

  return {
    headerText: "text-slate-400",
    cardHover: "hover:border-slate-500/30",
    arrowHover: "group-hover:text-slate-300",
  };
}

export default function TheoryPage() {
  const [searchParams] = useSearchParams();
  const activePropertyId = searchParams.get("property");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activePropertyId]);

  const isMain = !activePropertyId;
  const spec = activePropertyId ? theoryByPropertyId[activePropertyId] : null;

  if (activePropertyId && !spec) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-[#0f172a] p-10 font-mono text-blue-400">
        <div className="mb-4 animate-pulse text-2xl font-black uppercase">
          Property Not Found
        </div>
        <div className="text-sm text-slate-500">
          data/theory/properties/index.js에 "{activePropertyId}" 스펙이 없습니다.
        </div>
        <Link
          to="/compute/theory"
          className="mt-6 rounded-xl bg-blue-600 px-4 py-2 font-bold text-white"
        >
          Back to Property Atlas
        </Link>
      </div>
    );
  }

  return (
    <div className="flex min-h-dvh overflow-x-hidden bg-[#0f172a] text-slate-200 antialiased">
      <header className="fixed left-0 right-0 top-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur md:hidden">
        <div className="flex items-center justify-between px-5 py-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="rounded-xl bg-blue-600 p-2">
              <Cpu size={18} className="text-white" />
            </div>
            <div className="leading-none">
              <div className="font-black tracking-tight text-blue-400">
                AICF LAB
              </div>
              <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                v1.1.0 Property Atlas
              </div>
            </div>
          </Link>

          <button
            onClick={() => setIsSidebarOpen(true)}
            className="rounded-xl border border-slate-700 bg-[#1e293b] p-2"
            aria-label="Open sidebar"
          >
            <Menu size={18} />
          </button>
        </div>
      </header>

      <ComputeSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.1.0 Property Atlas"
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="h-[68px] md:hidden" />

        <div className="flex-1 space-y-14 overflow-y-auto bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))] p-6 pb-32 sm:p-10">
          {isMain ? (
            <div className="mx-auto max-w-6xl animate-in space-y-20 fade-in duration-700">
              <section className="relative overflow-hidden rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-10 shadow-2xl sm:p-16">
                <div className="pointer-events-none absolute -right-10 -top-10 text-[140px] font-black text-blue-500/5">
                  ATLAS
                </div>

                <div className="mb-6 flex items-center gap-2 font-mono text-xs font-black uppercase tracking-[0.3em] text-blue-500">
                  <BookOpen size={16} /> Property Atlas
                </div>

                <h1 className="text-4xl font-black leading-tight tracking-tight text-white sm:text-6xl">
                  Semantic Properties <br />
                  <span className="text-3xl text-blue-500 sm:text-5xl">
                    for Runtime Transformation
                  </span>
                </h1>

                <p className="mt-8 max-w-4xl text-lg leading-relaxed text-slate-400 sm:text-xl">
                  Property Atlas는 개별 operator 설명 페이지가 아니라,{" "}
                  <strong>
                    runtime transformation을 허용하는 수학적 / 논리적 property
                  </strong>
                  를 다루는 계층입니다.
                  <br />
                  이 페이지는 어떤 재배치, 분해, 병합, 타일링, 특수화가 의미
                  보존 아래에서 합법적인지 그 조건을 정리합니다.
                </p>

                <div className="mt-8 inline-flex items-center gap-2 rounded-2xl border border-blue-500/20 bg-blue-500/5 px-4 py-2 text-[11px] font-bold uppercase tracking-widest text-blue-300">
                  <ShieldCheck size={14} />
                  Algebraic Conditions for Legal Transformation
                </div>
              </section>

              <section className="grid grid-cols-1 gap-8 md:grid-cols-2">
                <div className="rounded-[2rem] border border-blue-500/20 bg-[#0b1120] p-8">
                  <div className="mb-6 flex items-center gap-3 text-blue-400">
                    <Target size={24} />
                    <h2 className="text-xl font-black uppercase">
                      What Property Atlas Covers
                    </h2>
                  </div>

                  <ul className="space-y-4 text-slate-300">
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
                      <span>
                        <strong>Foundational Properties:</strong> 어떤 algebraic
                        / semantic law가 transform legality를 직접 규정하는지
                        다룹니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
                      <span>
                        <strong>Structural Properties:</strong> 상위 semantic
                        property가 성립할 때 어떤 decomposition / accumulation /
                        tiling form이 가능한지 다룹니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
                      <span>
                        <strong>Legality Conditions:</strong> 어떤 조건 아래에서
                        transform이 의미 보존적으로 성립하는지 정리합니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-blue-500" />
                      <span>
                        <strong>Representative Realizations:</strong> 실제
                        construction, operator, subgraph 예시를 제공합니다.
                      </span>
                    </li>
                  </ul>
                </div>

                <div className="rounded-[2rem] border border-red-500/10 bg-[#0b1120] p-8">
                  <div className="mb-6 flex items-center gap-3 text-red-400">
                    <XCircle size={24} />
                    <h2 className="text-xl font-black uppercase text-slate-300">
                      What It Does NOT Focus On
                    </h2>
                  </div>

                  <ul className="space-y-4 italic text-slate-500">
                    <li className="flex gap-3 decoration-slate-700 line-through">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-900/50" />
                      <span>개별 operator의 상세 구현 walkthrough</span>
                    </li>
                    <li className="flex gap-3 decoration-slate-700 line-through">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-900/50" />
                      <span>커널 내부 CTA / warp scheduling 미세 최적화</span>
                    </li>
                    <li className="flex gap-3 decoration-slate-700 line-through">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-900/50" />
                      <span>벤치마크 수치, profiler metrics, hardware tuning</span>
                    </li>
                  </ul>

                  <p className="mt-10 rounded-xl border border-slate-800 bg-slate-900/50 p-4 text-[12px] font-medium leading-relaxed text-slate-400">
                    <span className="mb-1 block font-bold text-blue-400">
                      NOTE: Relationship with Ops Explorer
                    </span>
                    Property Atlas는 <strong>property 자체</strong>를 정의하고,
                    Ops Explorer는 각 operator가 어떤{" "}
                    <strong>property profile</strong>을 가지며 어떤 lowering
                    family로 이어지는지를 보여줍니다.
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

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
                  {[
                    {
                      id: "01",
                      title: "Definition",
                      desc: "property의 의미와 상태 공간 정의",
                      icon: Binary,
                    },
                    {
                      id: "02",
                      title: "Legality",
                      desc: "언제 transform이 합법적인가",
                      icon: ShieldCheck,
                    },
                    {
                      id: "03",
                      title: "Consequence",
                      desc: "허용되는 runtime / lowering 결과",
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
                      title: "Realization",
                      desc: "대표 construction / operator 예시",
                      icon: Boxes,
                    },
                  ].map((item) => (
                    <div
                      key={item.id}
                      className="group rounded-[2rem] border border-slate-800 bg-[#1e293b] p-8 shadow-lg transition hover:border-blue-500/30"
                    >
                      <item.icon
                        className="mb-6 text-blue-500 transition-transform group-hover:scale-110"
                        size={28}
                      />
                      <div className="mb-1 font-mono text-[10px] font-bold text-slate-600">
                        SECTION {item.id}
                      </div>
                      <h3 className="mb-2 text-lg font-black uppercase text-white">
                        {item.title}
                      </h3>
                      <p className="text-sm leading-relaxed text-slate-400">
                        {item.desc}
                      </p>
                    </div>
                  ))}
                </div>
              </section>

              {quickProperties.length > 0 && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-purple-400">
                    <ArrowRightLeft size={24} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                      Quick Entry Points
                    </h2>
                  </div>

                  <div className="flex flex-wrap gap-3">
                    {quickProperties
                      .filter((id) => theoryByPropertyId[id])
                      .map((id) => (
                        <Link
                          key={id}
                          to={`/compute/theory?property=${id}`}
                          className="rounded-2xl border border-slate-800 bg-[#1e293b] px-4 py-3 text-xs font-black uppercase tracking-wider text-slate-200 transition hover:border-purple-500/40 hover:text-white"
                        >
                          {theoryByPropertyId[id].title}
                        </Link>
                      ))}
                  </div>
                </section>
              )}

              {theoryPropertyGroups.map((group) => {
                const theme = getGroupTheme(group.id);

                return (
                  <section key={group.id} className="space-y-8">
                    <div className={`flex items-center gap-3 ${theme.headerText}`}>
                      <Workflow size={24} />
                      <div className="space-y-2">
                        <div>
                          <GroupBadge groupId={group.id} />
                        </div>
                        <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                          {group.title}
                        </h2>
                        <p className="max-w-3xl text-sm leading-relaxed text-slate-400">
                          {group.description}
                        </p>
                      </div>
                    </div>

                    <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
                      {group.items.map((item) => (
                        <Link
                          key={item.id}
                          to={`/compute/theory?property=${item.id}`}
                          className={`group rounded-[2rem] border border-slate-800 bg-[#1e293b] p-6 shadow-xl transition ${theme.cardHover}`}
                        >
                          <div className="mb-3 flex items-center justify-between gap-3">
                            <div className="text-[10px] font-black uppercase tracking-widest text-blue-500">
                              {item.subtitle || "Compute Property"}
                            </div>
                            <GroupBadge groupId={item.group} />
                          </div>

                          <div className="flex items-start justify-between gap-4">
                            <div className="min-w-0">
                              <h3 className="break-words text-xl font-black uppercase tracking-tight text-white">
                                {item.title}
                              </h3>
                              <p className="mt-3 text-sm leading-relaxed text-slate-400">
                                {item.hero?.lead ||
                                  "Algebraic or semantic condition that enables invariant-preserving runtime transformation."}
                              </p>
                            </div>
                            <ArrowRight
                              size={18}
                              className={`shrink-0 text-slate-600 transition ${theme.arrowHover}`}
                            />
                          </div>
                        </Link>
                      ))}
                    </div>
                  </section>
                );
              })}

              <section className="rounded-[3rem] border border-blue-500/20 bg-blue-600/5 p-12">
                <div className="max-w-4xl">
                  <div className="mb-4 text-[11px] font-black uppercase tracking-widest text-blue-400">
                    Core Principle
                  </div>

                  <h2 className="mb-5 text-2xl font-black uppercase leading-tight text-white sm:text-3xl">
                    Property explains why. Transform explains what.
                  </h2>

                  <p className="text-base leading-relaxed text-slate-400 sm:text-lg">
                    AICF에서 optimization은 operator 이름으로 정의되지 않고,
                    연산이 가진 semantic / algebraic property 위에서 정의됩니다.
                    property는 <strong>왜 바꿔도 되는가</strong>를 설명하고,
                    transform은 <strong>무엇을 바꾸는가</strong>를 설명합니다.
                  </p>
                </div>
              </section>

              <section className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-10 text-center">
                <div className="mb-4 flex items-center justify-center gap-2 text-emerald-400">
                  <ArrowRight size={18} />
                  <span className="text-[11px] font-black uppercase tracking-widest">
                    Next Layer
                  </span>
                </div>

                <h2 className="mb-4 text-2xl font-black uppercase text-white">
                  See How These Properties Appear in Real Operators
                </h2>

                <p className="mx-auto mb-8 max-w-2xl leading-relaxed text-slate-400">
                  Property Atlas가 property를 정의했다면, Ops Explorer는 각
                  operator가 어떤 property 조합을 가지며 어떤 lowering family로
                  이어지는지 보여줍니다.
                </p>

                <Link
                  to="/compute/ops"
                  className="inline-flex items-center gap-2 rounded-2xl border border-blue-500/20 bg-blue-600/10 px-8 py-4 font-black uppercase tracking-widest text-blue-300 transition hover:bg-blue-600/20"
                >
                  Go to Ops Explorer <ArrowRight size={16} />
                </Link>
              </section>
            </div>
          ) : (
            <div className="animate-in space-y-14 slide-in-from-bottom-4 duration-500">
              <section className="relative overflow-hidden rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-10 shadow-2xl sm:p-12">
                <div className="pointer-events-none absolute -right-10 -top-10 text-[120px] font-black uppercase tracking-tighter text-blue-500/5 sm:text-[160px]">
                  {spec.id}
                </div>

                <div className="mb-4 flex items-center gap-3">
                  <div className="font-mono text-[10px] font-black uppercase tracking-[0.35em] text-blue-500">
                    <div className="flex items-center gap-2">
                      <Waypoints size={14} />{" "}
                      {spec.subtitle || "Compute Property"}
                    </div>
                  </div>
                  <GroupBadge groupId={spec.group} />
                </div>

                <h1 className="mt-4 text-4xl font-black leading-[1.05] tracking-tight text-white sm:text-6xl">
                  {spec.title}
                </h1>

                {spec.hero?.lead && (
                  <p className="mt-6 max-w-4xl text-base italic leading-relaxed text-slate-400 sm:text-lg">
                    {spec.hero.lead}
                  </p>
                )}

                {spec.hero?.canonicalLatex && (
                  <div className="mt-8 overflow-x-auto rounded-3xl border border-slate-800/50 bg-[#0b1120] p-6 shadow-inner scrollbar-hide sm:p-8">
                    <div className="w-max min-w-full text-center text-2xl text-blue-300 sm:text-3xl">
                      <BlockMath math={spec.hero.canonicalLatex} />
                    </div>
                  </div>
                )}
              </section>

              {spec.sections?.definition && (
                <section className="grid grid-cols-1 items-start gap-8 lg:grid-cols-12">
                  <div className="min-w-0 space-y-6 lg:col-span-5">
                    <div className="flex items-center gap-3 text-purple-400">
                      <Shrink size={22} />
                      <h2 className="text-2xl font-black uppercase tracking-tight">
                        Definition
                      </h2>
                    </div>

                    <div className="space-y-3 leading-relaxed text-slate-400">
                      {spec.sections.definition.bullets?.map((b, i) => (
                        <div key={i} className="flex gap-3">
                          <div className="mt-2 h-1.5 w-1.5 shrink-0 rounded-full bg-purple-500" />
                          <div>
                            <span className="font-bold text-slate-200">
                              {b.k}
                            </span>{" "}
                            — {b.v}
                          </div>
                        </div>
                      ))}
                    </div>

                    {spec.sections.definition.latex && (
                      <div className="min-w-0 overflow-x-auto rounded-3xl border border-slate-800 bg-[#1e293b] p-6 shadow-xl scrollbar-hide">
                        <div className="w-max min-w-full">
                          <BlockMath math={spec.sections.definition.latex} />
                        </div>
                        <p className="mt-4 text-center text-[11px] font-bold uppercase tracking-widest text-slate-500">
                          Canonical Property Form
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="min-w-0 rounded-[2.5rem] border border-blue-500/20 bg-[#0b1120] p-8 lg:col-span-7">
                    <div className="mb-4 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-slate-500">
                      <Info size={12} /> Consequence View
                    </div>

                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                      {spec.sections.definition.preview?.map((r, i) => (
                        <div
                          key={i}
                          className="rounded-2xl border border-slate-800 bg-[#0f172a] p-5 transition hover:border-blue-500/30"
                        >
                          <div className="font-black uppercase tracking-tight text-slate-200">
                            {r.k}
                          </div>
                          <p className="mt-2 text-sm leading-relaxed text-slate-400">
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

                  <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                    {spec.sections.legality.cards?.map((c) => (
                      <div
                        key={c.id}
                        className="group min-w-0 rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8 shadow-xl transition hover:border-emerald-500/30"
                      >
                        <div className="mb-6 flex items-center justify-between">
                          <IconBadge icon={c.icon} />
                          <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-slate-600">
                            Rule {c.id}
                          </span>
                        </div>

                        <h3 className="mb-3 text-xl font-black uppercase italic tracking-tighter text-white">
                          {c.title}
                        </h3>

                        <p className="mb-5 text-sm leading-relaxed text-slate-400">
                          {c.desc}
                        </p>

                        {c.metric && (
                          <div className="min-w-0 rounded-2xl border border-slate-800 bg-[#0f172a] p-4 font-mono text-[11px] italic text-blue-200/70">
                            <div className="overflow-x-auto scrollbar-hide">
                              <div className="w-max">
                                <InlineMath math={c.metric} />
                              </div>
                            </div>
                          </div>
                        )}

                        {c.note && (
                          <div className="mt-4 text-[11px] font-bold uppercase tracking-widest text-emerald-400">
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
                      Transform Consequences
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                    {spec.sections.enables.items?.map((item, i) => (
                      <div
                        key={i}
                        className="rounded-[2rem] border border-slate-800 bg-[#1e293b] p-6"
                      >
                        <div className="mb-2 text-[10px] font-black uppercase tracking-widest text-blue-500">
                          Consequence {String(i + 1).padStart(2, "0")}
                        </div>
                        <div className="text-lg font-black uppercase tracking-tight text-white">
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

                  <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
                    {spec.sections.boundary.items?.map((item, i) => (
                      <div
                        key={i}
                        className="rounded-2xl border border-slate-800 bg-[#0f172a] p-6"
                      >
                        <div className="mb-2 text-[10px] font-black uppercase tracking-widest text-amber-400">
                          Boundary {String(i + 1).padStart(2, "0")}
                        </div>
                        <p className="text-sm leading-relaxed text-slate-400">
                          {item}
                        </p>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              <section className="grid grid-cols-1 gap-8 lg:grid-cols-2">
                {spec.sections?.relatedConstructions && (
                  <div className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8">
                    <div className="mb-6 flex items-center gap-3 text-purple-400">
                      <Boxes size={22} />
                      <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                        Representative Realizations
                      </h2>
                    </div>

                    <div className="flex flex-wrap gap-3">
                      {spec.sections.relatedConstructions.items?.map(
                        (item, i) => (
                          <ConstructionChip key={i} item={item} />
                        )
                      )}
                    </div>
                  </div>
                )}

                {spec.sections?.relatedTransforms && (
                  <div className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8">
                    <div className="mb-6 flex items-center gap-3 text-emerald-400">
                      <GitMerge size={22} />
                      <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                        Runtime / Lowering Consequences
                      </h2>
                    </div>

                    <div className="space-y-3">
                      {spec.sections.relatedTransforms.items?.map((t, i) => (
                        <div
                          key={i}
                          className="rounded-2xl border border-slate-800 bg-[#0f172a] p-4 text-sm font-bold text-slate-300"
                        >
                          {t}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </section>

              <div className="flex flex-col items-center justify-center gap-4 border-t border-slate-800 pt-10 sm:flex-row">
                <Link
                  to="/compute/theory"
                  className="flex items-center gap-2 text-sm font-black uppercase text-blue-400 transition hover:text-white"
                >
                  <BookOpen size={16} /> Back to Property Atlas
                </Link>

                <Link
                  to="/compute/ops"
                  className="flex items-center gap-2 text-sm font-black uppercase text-emerald-400 transition hover:text-white"
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