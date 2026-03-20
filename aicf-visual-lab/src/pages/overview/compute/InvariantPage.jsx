import React, { useEffect, useMemo, useState } from "react";
import "katex/dist/katex.min.css";
import { BlockMath, InlineMath } from "react-katex";
import { Link, useSearchParams } from "react-router-dom";
import {
  Cpu,
  Menu,
  ShieldCheck,
  BookOpen,
  ArrowRight,
  Workflow,
  Scale,
  Target,
  XCircle,
  Layers,
  Info,
  Waypoints,
  Lock,
  Boxes,
  GitMerge,
  Orbit,
  Gauge,
  Binary,
  Zap,
  CheckCircle2,
} from "lucide-react";

import ComputeSidebar from "../../../components/layout/ComputeSidebar.jsx";
import {
  theoryInvariantGroups,
  theoryByInvariantId,
  theoryIdToInvariantProfileKey,
} from "../../../data/invariants/index.js";
import { opsByInvariant, allOpsData } from "../../../data/ops/index.js";

const quickInvariants = [
  "SemanticConsistency",
  "NumericStability",
  "StructuralPreservation",
];

const iconMap = {
  shield: ShieldCheck,
  lock: Lock,
  scale: Scale,
  orbit: Orbit,
  gauge: Gauge,
  binary: Binary,
  zap: Zap,
  target: Target,
};

const INVARIANT_STATUS_TONE = {
  strong: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300",
  medium: "border-blue-500/20 bg-blue-500/5 text-blue-300",
  conditional: "border-amber-500/20 bg-amber-500/5 text-amber-300",
  limited: "border-purple-500/20 bg-purple-500/5 text-purple-300",
  weak: "border-slate-500/20 bg-slate-500/5 text-slate-300",
  not_applicable: "border-rose-500/20 bg-rose-500/5 text-rose-300",
};

function toInvariantProfileKey(invariantId) {
  return theoryIdToInvariantProfileKey[invariantId] ?? invariantId;
}

function formatScore(score) {
  if (typeof score !== "number") return null;
  return Math.round(score * 100);
}

function IconBadge({ icon }) {
  const Icon = iconMap[icon] ?? ShieldCheck;
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#0f172a] p-3">
      <Icon size={18} className="text-purple-400" />
    </div>
  );
}

function GroupBadge({ groupId }) {
  const meta =
    groupId === "semantic"
      ? {
          label: "Semantic",
          cls: "border-blue-500/20 bg-blue-500/5 text-blue-300",
        }
      : groupId === "numeric"
      ? {
          label: "Numeric",
          cls: "border-purple-500/20 bg-purple-500/5 text-purple-300",
        }
      : groupId === "structural"
      ? {
          label: "Structural",
          cls: "border-amber-500/20 bg-amber-500/5 text-amber-300",
        }
      : groupId === "stateful"
      ? {
          label: "Stateful",
          cls: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300",
        }
      : {
          label: "Invariant Group",
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
  if (groupId === "semantic") {
    return {
      headerText: "text-blue-400",
      cardHover: "hover:border-blue-500/30",
      arrowHover: "group-hover:text-blue-400",
    };
  }

  if (groupId === "numeric") {
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

  if (groupId === "stateful") {
    return {
      headerText: "text-emerald-400",
      cardHover: "hover:border-emerald-500/30",
      arrowHover: "group-hover:text-emerald-400",
    };
  }

  return {
    headerText: "text-slate-400",
    cardHover: "hover:border-slate-500/30",
    arrowHover: "group-hover:text-slate-300",
  };
}

function ConstructionChip({ item }) {
  if (typeof item === "string") {
    return (
      <span className="rounded-xl border border-slate-700 bg-[#0f172a] px-4 py-2 text-xs font-black uppercase tracking-wider text-purple-300">
        {item}
      </span>
    );
  }

  if (item?.op) {
    return (
      <Link
        to={`/compute/ops?op=${item.op}`}
        className="rounded-xl border border-slate-700 bg-[#0f172a] px-4 py-2 text-xs font-black uppercase tracking-wider text-purple-300 transition hover:border-purple-500"
      >
        {item.label ?? item.op}
      </Link>
    );
  }

  return (
    <span className="rounded-xl border border-slate-700 bg-[#0f172a] px-4 py-2 text-xs font-black uppercase tracking-wider text-purple-300">
      {item?.label ?? "Unknown"}
    </span>
  );
}

function InvariantMatchCard({ match }) {
  const op = allOpsData[match.opId];
  if (!op) return null;

  const score = formatScore(match.score);
  const toneClass =
    INVARIANT_STATUS_TONE[match.status] ?? INVARIANT_STATUS_TONE.weak;

  return (
    <Link
      to={`/compute/ops?op=${match.opId}`}
      className="group rounded-[2rem] border border-slate-800 bg-[#1e293b] p-6 shadow-xl transition hover:border-purple-500/30"
    >
      <div className="mb-3 flex items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="text-[10px] font-black uppercase tracking-widest text-purple-500">
            Operator Match
          </div>
          <h3 className="mt-2 break-words text-xl font-black uppercase tracking-tight text-white">
            {match.opId}
          </h3>
        </div>

        <ArrowRight
          size={18}
          className="shrink-0 text-slate-600 transition group-hover:text-purple-400"
        />
      </div>

      <p className="mb-4 text-sm leading-relaxed text-slate-400">
        {op.category || "Uncategorized"}
      </p>

      <div className="mb-4 flex items-center gap-2">
        <span
          className={`rounded-xl border px-3 py-1 text-[10px] font-black uppercase tracking-widest ${toneClass}`}
        >
          {match.status?.replaceAll("_", " ") || "unknown"}
        </span>

        {score !== null && (
          <span className="rounded-xl border border-slate-700 bg-[#0f172a] px-3 py-1 text-[10px] font-black uppercase tracking-widest text-slate-300">
            Affinity {score}
          </span>
        )}
      </div>

      {score !== null && (
        <div className="mb-4">
          <div className="mb-2 flex items-center justify-between text-[9px] font-black uppercase tracking-widest text-slate-500">
            <span>Invariant Affinity</span>
            <span>{score}</span>
          </div>
          <div className="h-2 overflow-hidden rounded-full border border-slate-800 bg-[#0f172a]">
            <div
              className="h-full bg-purple-400"
              style={{ width: `${score}%` }}
            />
          </div>
        </div>
      )}

      <p className="text-sm leading-relaxed text-slate-500">
        {op.descriptions?.oneLine ||
          op.descriptions?.essence ||
          "Invariant profile available in Ops Explorer."}
      </p>
    </Link>
  );
}

export default function InvariantPage() {
  const [searchParams] = useSearchParams();
  const activeInvariantId = searchParams.get("invariant");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activeInvariantId]);

  const isMain = !activeInvariantId;
  const spec = activeInvariantId ? theoryByInvariantId[activeInvariantId] : null;

  const matchedOps = useMemo(() => {
    if (!activeInvariantId) return [];
    const invariantKey = toInvariantProfileKey(activeInvariantId);
    return opsByInvariant?.[invariantKey] ?? [];
  }, [activeInvariantId]);

  if (activeInvariantId && !spec) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-[#0f172a] p-10 font-mono text-purple-400">
        <div className="mb-4 animate-pulse text-2xl font-black uppercase">
          Invariant Not Found
        </div>
        <div className="text-sm text-slate-500">
          data/theory/invariants index에 "{activeInvariantId}" 스펙이 없습니다.
        </div>
        <Link
          to="/compute/invariants"
          className="mt-6 rounded-xl bg-purple-600 px-4 py-2 font-bold text-white"
        >
          Back to Invariant Atlas
        </Link>
      </div>
    );
  }

  return (
    <div className="flex min-h-dvh overflow-x-hidden bg-[#0f172a] text-slate-200 antialiased">
      <header className="fixed left-0 right-0 top-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur md:hidden">
        <div className="flex items-center justify-between px-5 py-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="rounded-xl bg-purple-600 p-2">
              <Cpu size={18} className="text-white" />
            </div>
            <div className="leading-none">
              <div className="font-black tracking-tight text-purple-400">
                AICF LAB
              </div>
              <div className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                v1.1.0 Invariant Atlas
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
        version="v1.1.0 Invariant Atlas"
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="h-[68px] md:hidden" />

        <div className="flex-1 space-y-14 overflow-y-auto bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))] p-6 pb-32 sm:p-10">
          {isMain ? (
            <div className="mx-auto max-w-6xl animate-in space-y-20 fade-in duration-700">
              <section className="relative overflow-hidden rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-10 shadow-2xl sm:p-16">
                <div className="pointer-events-none absolute -right-10 -top-10 text-[140px] font-black text-purple-500/5">
                  ATLAS
                </div>

                <div className="mb-6 flex items-center gap-2 font-mono text-xs font-black uppercase tracking-[0.3em] text-purple-500">
                  <BookOpen size={16} /> Invariant Atlas
                </div>

                <h1 className="text-4xl font-black leading-tight tracking-tight text-white sm:text-6xl">
                  Semantic Invariants <br />
                  <span className="text-3xl text-purple-500 sm:text-5xl">
                    for Runtime Safety and Realization Boundaries
                  </span>
                </h1>

                <p className="mt-8 max-w-4xl text-lg leading-relaxed text-slate-400 sm:text-xl">
                  Invariant Atlas는 어떤 transform이 가능한지를 설명하는 페이지가 아니라,{" "}
                  <strong>transform 이후에도 반드시 유지되어야 하는 의미적 / 수치적 / 구조적 조건</strong>
                  을 다루는 계층입니다.
                  <br />
                  이 페이지는 runtime path selection, lowering, approximation,
                  fusion, tiling 이후에도 무엇이 깨지면 안 되는지를 정리합니다.
                </p>

                <div className="mt-8 inline-flex items-center gap-2 rounded-2xl border border-purple-500/20 bg-purple-500/5 px-4 py-2 text-[11px] font-bold uppercase tracking-widest text-purple-300">
                  <Lock size={14} />
                  Conditions That Must Survive Transformation
                </div>
              </section>

              <section className="grid grid-cols-1 gap-8 md:grid-cols-2">
                <div className="rounded-[2rem] border border-purple-500/20 bg-[#0b1120] p-8">
                  <div className="mb-6 flex items-center gap-3 text-purple-400">
                    <Target size={24} />
                    <h2 className="text-xl font-black uppercase">
                      What Invariant Atlas Covers
                    </h2>
                  </div>

                  <ul className="space-y-4 text-slate-300">
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-purple-500" />
                      <span>
                        <strong>Semantic Invariants:</strong> 출력 meaning, contract,
                        interpretation이 보존되어야 하는 조건을 다룹니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-purple-500" />
                      <span>
                        <strong>Numeric Invariants:</strong> stability, bounded error,
                        normalization safety를 다룹니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-purple-500" />
                      <span>
                        <strong>Structural Invariants:</strong> shape relation,
                        dependency structure, reduction contract 보존을 다룹니다.
                      </span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-purple-500" />
                      <span>
                        <strong>Runtime Guard Meaning:</strong> runtime이 어떤
                        조건을 검사해야 안전한지 정리합니다.
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
                      <span>허용 transform 목록 자체의 taxonomy</span>
                    </li>
                    <li className="flex gap-3 decoration-slate-700 line-through">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-900/50" />
                      <span>개별 kernel micro scheduling 디테일</span>
                    </li>
                    <li className="flex gap-3 decoration-slate-700 line-through">
                      <div className="mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-red-900/50" />
                      <span>단순 benchmark 수치 비교</span>
                    </li>
                  </ul>

                  <p className="mt-10 rounded-xl border border-slate-800 bg-slate-900/50 p-4 text-[12px] font-medium leading-relaxed text-slate-400">
                    <span className="mb-1 block font-bold text-purple-400">
                      NOTE: Relationship with Property Atlas
                    </span>
                    Property Atlas는 <strong>무엇이 허용되는가</strong>를 정의하고,
                    Invariant Atlas는 <strong>무엇이 반드시 유지되어야 하는가</strong>를 정의합니다.
                    Runtime은 둘의 교집합 안에서만 경로를 선택해야 합니다.
                  </p>
                </div>
              </section>

              <section className="space-y-8">
                <div className="flex items-center gap-3 text-blue-400">
                  <Layers size={24} />
                  <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                    Invariant Spec Structure
                  </h2>
                </div>

                <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-5">
                  {[
                    {
                      id: "01",
                      title: "Meaning",
                      desc: "무엇이 변하면 안 되는가",
                      icon: Lock,
                    },
                    {
                      id: "02",
                      title: "Guard",
                      desc: "runtime이 무엇을 검사해야 하는가",
                      icon: ShieldCheck,
                    },
                    {
                      id: "03",
                      title: "Preserves",
                      desc: "보존되어야 하는 핵심 항목",
                      icon: CheckCircle2,
                    },
                    {
                      id: "04",
                      title: "Failure",
                      desc: "어떤 조건에서 invariant가 깨지는가",
                      icon: Scale,
                    },
                    {
                      id: "05",
                      title: "Consequence",
                      desc: "이 invariant가 제약하는 realization 범위",
                      icon: Zap,
                    },
                  ].map((item) => (
                    <div
                      key={item.id}
                      className="group rounded-[2rem] border border-slate-800 bg-[#1e293b] p-8 shadow-lg transition hover:border-purple-500/30"
                    >
                      <item.icon
                        className="mb-6 text-purple-500 transition-transform group-hover:scale-110"
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

              {quickInvariants.length > 0 && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-emerald-400">
                    <ArrowRight size={24} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                      Quick Entry Points
                    </h2>
                  </div>

                  <div className="flex flex-wrap gap-3">
                    {quickInvariants
                      .filter((id) => theoryByInvariantId[id])
                      .map((id) => (
                        <Link
                          key={id}
                          to={`/compute/invariants?invariant=${id}`}
                          className="rounded-2xl border border-slate-800 bg-[#1e293b] px-4 py-3 text-xs font-black uppercase tracking-wider text-slate-200 transition hover:border-purple-500/40 hover:text-white"
                        >
                          {theoryByInvariantId[id].title}
                        </Link>
                      ))}
                  </div>
                </section>
              )}

              {theoryInvariantGroups.map((group) => {
                const theme = getGroupTheme(group.id);

                if (!group.items?.length) return null;

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
                          to={`/compute/invariants?invariant=${item.id}`}
                          className={`group rounded-[2rem] border border-slate-800 bg-[#1e293b] p-6 shadow-xl transition ${theme.cardHover}`}
                        >
                          <div className="mb-3 flex items-center justify-between gap-3">
                            <div className="text-[10px] font-black uppercase tracking-widest text-purple-500">
                              {item.subtitle || "Compute Invariant"}
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
                                  "Invariant condition that must remain preserved under valid runtime realization."}
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

              <section className="rounded-[3rem] border border-purple-500/20 bg-purple-600/5 p-12">
                <div className="max-w-4xl">
                  <div className="mb-4 text-[11px] font-black uppercase tracking-widest text-purple-400">
                    Core Principle
                  </div>

                  <h2 className="mb-5 text-2xl font-black uppercase leading-tight text-white sm:text-3xl">
                    Property opens the space. Invariant bounds the space.
                  </h2>

                  <p className="text-base leading-relaxed text-slate-400 sm:text-lg">
                    AICF에서 optimization이 성립하려면 먼저 property가 변환 가능성을 열고,
                    그다음 invariant가 그 변환이 어디까지 허용되는지를 제한해야 합니다.
                    invariant는 부가 설명이 아니라 <strong>runtime safety condition</strong> 입니다.
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
                  See How Real Operators Carry These Invariants
                </h2>

                <p className="mx-auto mb-8 max-w-2xl leading-relaxed text-slate-400">
                  Invariant Atlas가 보존 조건을 정의했다면, Ops Explorer는 각
                  operator가 실제로 어떤 invariant를 강하게 요구하는지 보여줍니다.
                </p>

                <Link
                  to="/compute/ops"
                  className="inline-flex items-center gap-2 rounded-2xl border border-purple-500/20 bg-purple-600/10 px-8 py-4 font-black uppercase tracking-widest text-purple-300 transition hover:bg-purple-600/20"
                >
                  Go to Ops Explorer <ArrowRight size={16} />
                </Link>
              </section>
            </div>
          ) : (
            <div className="animate-in space-y-14 slide-in-from-bottom-4 duration-500">
              <section className="relative overflow-hidden rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-10 shadow-2xl sm:p-12">
                <div className="pointer-events-none absolute -right-10 -top-10 text-[120px] font-black uppercase tracking-tighter text-purple-500/5 sm:text-[160px]">
                  {spec.id}
                </div>

                <div className="mb-4 flex items-center gap-3">
                  <div className="font-mono text-[10px] font-black uppercase tracking-[0.35em] text-purple-500">
                    <div className="flex items-center gap-2">
                      <Waypoints size={14} />{" "}
                      {spec.subtitle || "Compute Invariant"}
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
                    <div className="w-max min-w-full text-center text-2xl text-purple-300 sm:text-3xl">
                      <BlockMath math={spec.hero.canonicalLatex} />
                    </div>
                  </div>
                )}
              </section>

              {spec.sections?.meaning && (
                <section className="grid grid-cols-1 items-start gap-8 lg:grid-cols-12">
                  <div className="min-w-0 space-y-6 lg:col-span-5">
                    <div className="flex items-center gap-3 text-purple-400">
                      <Lock size={22} />
                      <h2 className="text-2xl font-black uppercase tracking-tight">
                        Meaning
                      </h2>
                    </div>

                    <div className="space-y-3 leading-relaxed text-slate-400">
                      {spec.sections.meaning.bullets?.map((b, i) => (
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

                    {spec.sections.meaning.latex && (
                      <div className="min-w-0 overflow-x-auto rounded-3xl border border-slate-800 bg-[#1e293b] p-6 shadow-xl scrollbar-hide">
                        <div className="w-max min-w-full">
                          <BlockMath math={spec.sections.meaning.latex} />
                        </div>
                        <p className="mt-4 text-center text-[11px] font-bold uppercase tracking-widest text-slate-500">
                          Canonical Invariant Form
                        </p>
                      </div>
                    )}
                  </div>

                  <div className="min-w-0 rounded-[2.5rem] border border-purple-500/20 bg-[#0b1120] p-8 lg:col-span-7">
                    <div className="mb-4 flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-slate-500">
                      <Info size={12} /> Observable View
                    </div>

                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                      {spec.sections.meaning.preview?.map((r, i) => (
                        <div
                          key={i}
                          className="rounded-2xl border border-slate-800 bg-[#0f172a] p-5 transition hover:border-purple-500/30"
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

              {spec.sections?.guard && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-emerald-400">
                    <ShieldCheck size={22} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                      Guard Conditions
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 gap-6 md:grid-cols-3">
                    {spec.sections.guard.cards?.map((c) => (
                      <div
                        key={c.id}
                        className="group min-w-0 rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8 shadow-xl transition hover:border-emerald-500/30"
                      >
                        <div className="mb-6 flex items-center justify-between">
                          <IconBadge icon={c.icon} />
                          <span className="font-mono text-[10px] font-bold uppercase tracking-widest text-slate-600">
                            Guard {c.id}
                          </span>
                        </div>

                        <h3 className="mb-3 text-xl font-black uppercase italic tracking-tighter text-white">
                          {c.title}
                        </h3>

                        <p className="mb-5 text-sm leading-relaxed text-slate-400">
                          {c.desc}
                        </p>

                        {c.metric && (
                          <div className="min-w-0 rounded-2xl border border-slate-800 bg-[#0f172a] p-4 font-mono text-[11px] italic text-purple-200/70">
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

              {spec.sections?.preserves && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-blue-400">
                    <CheckCircle2 size={22} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                      What Must Be Preserved
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
                    {spec.sections.preserves.items?.map((item, i) => (
                      <div
                        key={i}
                        className="rounded-[2rem] border border-slate-800 bg-[#1e293b] p-6"
                      >
                        <div className="mb-2 text-[10px] font-black uppercase tracking-widest text-blue-500">
                          Preserve {String(i + 1).padStart(2, "0")}
                        </div>
                        <div className="text-lg font-black uppercase tracking-tight text-white">
                          {item}
                        </div>
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {spec.sections?.failure && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-amber-400">
                    <Scale size={22} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                      Failure Boundaries
                    </h2>
                  </div>

                  <div className="grid grid-cols-1 gap-5 md:grid-cols-3">
                    {spec.sections.failure.items?.map((item, i) => (
                      <div
                        key={i}
                        className="rounded-2xl border border-slate-800 bg-[#0f172a] p-6"
                      >
                        <div className="mb-2 text-[10px] font-black uppercase tracking-widest text-amber-400">
                          Failure {String(i + 1).padStart(2, "0")}
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
                      {spec.sections.relatedConstructions.items?.map((item, i) => (
                        <ConstructionChip key={i} item={item} />
                      ))}
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

              <section className="space-y-8">
                <div className="flex items-center gap-3 text-purple-400">
                  <Workflow size={22} />
                  <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                    Operators Matching This Invariant
                  </h2>
                </div>

                <p className="max-w-3xl text-sm leading-relaxed text-slate-400">
                  아래 operator들은 현재 registry 기준으로 이 invariant를 강하게
                  또는 의미 있게 요구하는 순서대로 정렬되어 있습니다.
                </p>

                {matchedOps.length > 0 ? (
                  <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
                    {matchedOps.map((match) => (
                      <InvariantMatchCard
                        key={`${activeInvariantId}-${match.opId}`}
                        match={match}
                      />
                    ))}
                  </div>
                ) : (
                  <div className="rounded-[2rem] border border-dashed border-slate-700 bg-[#1e293b] p-8 text-sm text-slate-500">
                    아직 이 invariant와 연결된 operator profile이 없습니다.
                  </div>
                )}
              </section>

              <div className="flex flex-col items-center justify-center gap-4 border-t border-slate-800 pt-10 sm:flex-row">
                <Link
                  to="/compute/invariants"
                  className="flex items-center gap-2 text-sm font-black uppercase text-purple-400 transition hover:text-white"
                >
                  <BookOpen size={16} /> Back to Invariant Atlas
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