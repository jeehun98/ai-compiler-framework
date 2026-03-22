import React, { useEffect, useMemo, useState } from "react";
import "katex/dist/katex.min.css";
import { BlockMath, InlineMath } from "react-katex";
import { Link, useSearchParams } from "react-router-dom";
import {
  Menu,
  ShieldCheck,
  BookOpen,
  ArrowRight,
  Workflow,
  Scale,
  Target,
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

const quickRules = [
  "ReductionEquivalence",
  "NormalizationPreservation",
  "DomainPruningPreservation",
  "TiledExecutionEquivalence",
  "RepresentationEquivalence",
  "BoundedNumericDrift",
  "DecisionTolerance",
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
  boxes: Boxes,
  layers: Layers,
  gitmerge: GitMerge,
  waypoints: Waypoints,
};

const RULE_STATUS_TONE = {
  strong: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300",
  medium: "border-blue-500/20 bg-blue-500/5 text-blue-300",
  conditional: "border-amber-500/20 bg-amber-500/5 text-amber-300",
  limited: "border-purple-500/20 bg-purple-500/5 text-purple-300",
  weak: "border-slate-500/20 bg-slate-500/5 text-slate-300",
  not_applicable: "border-rose-500/20 bg-rose-500/5 text-rose-300",
};

function toRuleProfileKey(ruleId) {
  return theoryIdToInvariantProfileKey[ruleId] ?? ruleId;
}

function formatScore(score) {
  if (typeof score !== "number") return null;
  return Math.round(score * 100);
}

function GroupBadge({ groupId }) {
  const groupMap = {
    execution: {
      label: "실행",
      cls: "border-blue-500/20 bg-blue-500/5 text-blue-300",
    },
    "execution-meaning": {
      label: "실행",
      cls: "border-blue-500/20 bg-blue-500/5 text-blue-300",
    },
    normalization: {
      label: "정규화",
      cls: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300",
    },
    "normalization-safety": {
      label: "정규화",
      cls: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300",
    },
    pruning: {
      label: "생략",
      cls: "border-amber-500/20 bg-amber-500/5 text-amber-300",
    },
    tiling: {
      label: "타일링",
      cls: "border-cyan-500/20 bg-cyan-500/5 text-cyan-300",
    },
    representation: {
      label: "표현",
      cls: "border-fuchsia-500/20 bg-fuchsia-500/5 text-fuchsia-300",
    },
    numeric: {
      label: "수치",
      cls: "border-purple-500/20 bg-purple-500/5 text-purple-300",
    },
    "numeric-safety": {
      label: "수치",
      cls: "border-purple-500/20 bg-purple-500/5 text-purple-300",
    },
    tolerance: {
      label: "허용 한계",
      cls: "border-rose-500/20 bg-rose-500/5 text-rose-300",
    },
    "downstream-aware": {
      label: "허용 한계",
      cls: "border-rose-500/20 bg-rose-500/5 text-rose-300",
    },
  };

  const meta = groupMap[groupId] ?? {
    label: "규칙",
    cls: "border-slate-500/20 bg-slate-500/5 text-slate-300",
  };

  return (
    <span
      className={`inline-flex items-center rounded-full border px-3 py-1 text-[10px] font-black tracking-widest ${meta.cls}`}
    >
      {meta.label}
    </span>
  );
}

function getGroupTheme(groupId) {
  const themeMap = {
    execution: {
      headerText: "text-blue-400",
      cardBorder: "hover:border-blue-500/30",
      chip: "border-blue-500/20 bg-blue-500/5 text-blue-300",
    },
    "execution-meaning": {
      headerText: "text-blue-400",
      cardBorder: "hover:border-blue-500/30",
      chip: "border-blue-500/20 bg-blue-500/5 text-blue-300",
    },
    normalization: {
      headerText: "text-emerald-400",
      cardBorder: "hover:border-emerald-500/30",
      chip: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300",
    },
    "normalization-safety": {
      headerText: "text-emerald-400",
      cardBorder: "hover:border-emerald-500/30",
      chip: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300",
    },
    pruning: {
      headerText: "text-amber-400",
      cardBorder: "hover:border-amber-500/30",
      chip: "border-amber-500/20 bg-amber-500/5 text-amber-300",
    },
    tiling: {
      headerText: "text-cyan-400",
      cardBorder: "hover:border-cyan-500/30",
      chip: "border-cyan-500/20 bg-cyan-500/5 text-cyan-300",
    },
    representation: {
      headerText: "text-fuchsia-400",
      cardBorder: "hover:border-fuchsia-500/30",
      chip: "border-fuchsia-500/20 bg-fuchsia-500/5 text-fuchsia-300",
    },
    numeric: {
      headerText: "text-purple-400",
      cardBorder: "hover:border-purple-500/30",
      chip: "border-purple-500/20 bg-purple-500/5 text-purple-300",
    },
    "numeric-safety": {
      headerText: "text-purple-400",
      cardBorder: "hover:border-purple-500/30",
      chip: "border-purple-500/20 bg-purple-500/5 text-purple-300",
    },
    tolerance: {
      headerText: "text-rose-400",
      cardBorder: "hover:border-rose-500/30",
      chip: "border-rose-500/20 bg-rose-500/5 text-rose-300",
    },
    "downstream-aware": {
      headerText: "text-rose-400",
      cardBorder: "hover:border-rose-500/30",
      chip: "border-rose-500/20 bg-rose-500/5 text-rose-300",
    },
  };

  return (
    themeMap[groupId] ?? {
      headerText: "text-slate-300",
      cardBorder: "hover:border-slate-500/30",
      chip: "border-slate-500/20 bg-slate-500/5 text-slate-300",
    }
  );
}

function IconBadge({ icon }) {
  const Icon = iconMap[icon] ?? ShieldCheck;
  return (
    <div className="rounded-2xl border border-slate-800 bg-[#0f172a] p-3">
      <Icon size={18} className="text-purple-400" />
    </div>
  );
}

function ConstructionChip({ item }) {
  const opId = item?.op ?? item?.id;
  const label = item?.label ?? item?.title ?? opId;

  if (!opId) {
    return (
      <div className="rounded-2xl border border-slate-800 bg-[#0f172a] px-4 py-3 text-xs font-black tracking-wider text-slate-300">
        {label}
      </div>
    );
  }

  return (
    <Link
      to={`/compute/ops?op=${opId}`}
      className="rounded-2xl border border-slate-800 bg-[#0f172a] px-4 py-3 text-xs font-black tracking-wider text-slate-300 transition hover:border-purple-500/40 hover:text-white"
    >
      {label}
    </Link>
  );
}

function RuleMatchCard({ match }) {
  const tone = RULE_STATUS_TONE[match.status] ?? RULE_STATUS_TONE.weak;
  const op = allOpsData?.[match.opId];

  return (
    <Link
      to={`/compute/ops?op=${match.opId}`}
      className="group rounded-[2rem] border border-slate-800 bg-[#1e293b] p-6 transition hover:border-purple-500/30"
    >
      <div className="mb-4 flex items-start justify-between gap-4">
        <div>
          <div className="text-[10px] font-black tracking-widest text-slate-500">
            연산
          </div>
          <h3 className="mt-1 text-lg font-black tracking-tight text-white">
            {op?.title ?? match.opId}
          </h3>
        </div>

        <span
          className={`rounded-full border px-3 py-1 text-[10px] font-black tracking-widest ${tone}`}
        >
          {(match.status ?? "weak").replaceAll("_", " ")}
        </span>
      </div>

      {typeof match.score === "number" && (
        <div className="mb-4 rounded-2xl border border-slate-800 bg-[#0f172a] p-4">
          <div className="mb-2 flex items-center justify-between text-[11px] font-bold tracking-widest text-slate-500">
            <span>연결 강도</span>
            <span>{formatScore(match.score)}%</span>
          </div>
          <div className="h-2 rounded-full bg-slate-800">
            <div
              className="h-2 rounded-full bg-gradient-to-r from-purple-500 to-emerald-400"
              style={{ width: `${formatScore(match.score)}%` }}
            />
          </div>
        </div>
      )}

      {match.reason && (
        <p className="text-sm leading-relaxed text-slate-400">{match.reason}</p>
      )}
    </Link>
  );
}

function MainHero() {
  return (
    <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-8 sm:p-10 lg:p-12 shadow-2xl">
      <div className="mb-6 flex items-center gap-2 text-blue-500 font-mono text-xs font-black tracking-[0.3em]">
        <BookOpen size={16} /> Preservation Guide
      </div>

      <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
        실행 의미를 유지하기 위한
        <br />
        <span className="text-blue-500 text-3xl sm:text-5xl">
          보존 규칙과 허용 경계
        </span>
      </h1>

      <p className="mt-8 max-w-3xl text-slate-400 text-lg sm:text-xl leading-relaxed font-light">
        이 페이지는 transform 이후에도 유지되어야 할 의미 조건,
        실사용에서 허용 가능한 변화 범위, 그리고 semantic failure로
        넘어가는 경계를 함께 다룹니다.
      </p>

      <div className="mt-8 inline-flex items-center gap-2 rounded-2xl border border-blue-500/20 bg-blue-500/5 px-4 py-2 text-[11px] font-bold tracking-widest text-blue-300">
        <ShieldCheck size={14} />
        Preservation / Tolerance View
      </div>
    </section>
  );
}

function MainStructureSection() {
  return (
    <section className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 sm:p-10">
      <div className="mb-6 flex items-center gap-3 text-emerald-400">
        <GitMerge size={22} />
        <h2 className="text-2xl font-black tracking-tight text-white">
          이 페이지의 구조
        </h2>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-3">
        {[
          {
            step: "01",
            title: "Preservation Rule",
            desc: "변환 이후에도 유지되어야 하는 의미 조건을 정의합니다.",
          },
          {
            step: "02",
            title: "Allowed Variation",
            desc: "실사용 관점에서 허용 가능한 변화 범위를 정의합니다.",
          },
          {
            step: "03",
            title: "Related Ops",
            desc: "어떤 연산이 해당 규칙과 강하게 연결되는지 보여줍니다.",
          },
        ].map((item) => (
          <div
            key={item.step}
            className="rounded-2xl border border-slate-800 bg-[#111827] p-5"
          >
            <div className="mb-2 text-[10px] font-black tracking-widest text-blue-500">
              Step {item.step}
            </div>
            <h3 className="mb-2 text-sm font-black text-white">{item.title}</h3>
            <p className="text-sm leading-relaxed text-slate-400">{item.desc}</p>
          </div>
        ))}
      </div>
    </section>
  );
}

function QuickEntrySection({ onSelectRule }) {
  return (
    <section className="space-y-8">
      <div className="flex items-center gap-3 text-emerald-400">
        <Workflow size={24} />
        <h2 className="text-2xl font-black tracking-tight text-white">
          빠른 진입점
        </h2>
      </div>

      <p className="max-w-3xl text-sm leading-relaxed text-slate-400">
        자주 참조되는 보존 규칙과 허용 경계부터 바로 들어갈 수 있도록 정리했습니다.
      </p>

      <div className="flex flex-wrap gap-4">
        {quickRules
          .filter((id) => theoryByInvariantId[id])
          .map((id) => (
            <button
              key={id}
              onClick={() => onSelectRule(id)}
              className="rounded-2xl border border-slate-700 bg-[#1e293b] px-6 py-3 text-xs font-bold tracking-wider text-blue-300 transition hover:border-blue-500"
            >
              {theoryByInvariantId[id].title}
            </button>
          ))}
      </div>
    </section>
  );
}

function RuleGroupSection({ group, onSelectRule }) {
  const theme = getGroupTheme(group.id);

  return (
    <section className="space-y-6">
      <div>
        <div className="mb-2">
          <GroupBadge groupId={group.id} />
        </div>
        <h2 className={`text-2xl font-black ${theme.headerText}`}>
          {group.title}
        </h2>
        {group.description && (
          <p className="mt-2 max-w-3xl text-sm leading-relaxed text-slate-400">
            {group.description}
          </p>
        )}
      </div>

      <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
        {group.items.map((item) => (
          <button
            key={item.id}
            onClick={() => onSelectRule(item.id)}
            className={`text-left bg-[#1e293b] border border-slate-800 rounded-[2rem] p-6 transition ${theme.cardBorder}`}
          >
            <div className="mb-2 text-[10px] font-black tracking-widest text-slate-500">
              규칙
            </div>
            <h3 className="text-xl font-black tracking-tight text-white">
              {item.title}
            </h3>
            {item.subtitle && (
              <p className="mt-2 text-sm font-bold text-slate-500">
                {item.subtitle}
              </p>
            )}
            <p className="mt-4 line-clamp-4 text-sm leading-relaxed text-slate-400">
              {item.hero?.lead}
            </p>
          </button>
        ))}
      </div>
    </section>
  );
}

function PreservationGuideView({ groups, onSelectRule }) {
  return (
    <div className="w-full max-w-6xl space-y-12">
      <MainHero />
      <MainStructureSection />
      <QuickEntrySection onSelectRule={onSelectRule} />
      {groups.map((group) => (
        <RuleGroupSection
          key={group.id}
          group={group}
          onSelectRule={onSelectRule}
        />
      ))}
    </div>
  );
}

function PreservationDetailView({ spec, matchedOps }) {
  return (
    <div className="animate-in slide-in-from-bottom-4 duration-500 space-y-8">
      <section className="flex flex-col justify-between gap-6 border-b border-slate-800 pb-6 lg:flex-row lg:items-end">
        <div className="min-w-0 space-y-2">
          <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black tracking-[0.3em]">
            <Lock size={14} /> Rule Report
          </div>
          <h2 className="break-words text-4xl font-black tracking-tight text-white leading-tight sm:text-6xl">
            {spec.title}
          </h2>
          {spec.subtitle && (
            <p className="text-sm font-bold text-slate-500">{spec.subtitle}</p>
          )}
        </div>

        <div className="flex w-fit items-center gap-2 rounded-xl border border-emerald-400/10 bg-emerald-400/5 px-4 py-2 text-[11px] font-bold tracking-widest text-emerald-400">
          <ShieldCheck size={16} /> Preservation Anchored
        </div>
      </section>

      {spec.hero?.lead && (
        <section className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-6 sm:p-8">
          <p className="max-w-4xl text-base leading-relaxed text-slate-300">
            {spec.hero.lead}
          </p>

          {spec.hero.canonicalLatex && (
            <div className="mt-6 overflow-x-auto rounded-2xl border border-slate-800 bg-[#0f172a] p-5 scrollbar-hide">
              <div className="w-max min-w-full text-purple-200/90">
                <BlockMath math={spec.hero.canonicalLatex} />
              </div>
            </div>
          )}
        </section>
      )}

      {spec.sections?.meaning && (
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-blue-400">
            <Lock size={22} />
            <h2 className="text-2xl font-black tracking-tight text-white">
              보존 의미
            </h2>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-12">
            <div className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8 lg:col-span-5">
              <div className="mb-4 flex items-center gap-2 text-[10px] font-black tracking-widest text-slate-500">
                <BookOpen size={12} /> 핵심 규칙
              </div>

              {spec.sections.meaning.bullets?.length > 0 && (
                <div className="space-y-4">
                  {spec.sections.meaning.bullets.map((b, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-slate-800 bg-[#0f172a] p-5"
                    >
                      <div className="font-black tracking-tight text-white">
                        {b.k}
                      </div>
                      <p className="mt-2 text-sm leading-relaxed text-slate-400">
                        {b.v}
                      </p>
                    </div>
                  ))}
                </div>
              )}

              {spec.sections.meaning.latex && (
                <div className="mt-6 rounded-2xl border border-slate-800 bg-[#0b1120] p-4 font-mono text-purple-200/80">
                  <div className="overflow-x-auto scrollbar-hide">
                    <div className="w-max">
                      <InlineMath math={spec.sections.meaning.latex} />
                    </div>
                  </div>
                </div>
              )}
            </div>

            <div className="rounded-[2.5rem] border border-purple-500/20 bg-[#0b1120] p-8 lg:col-span-7">
              <div className="mb-4 flex items-center gap-2 text-[10px] font-black tracking-widest text-slate-500">
                <Info size={12} /> 해석 관점
              </div>

              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {spec.sections.meaning.preview?.map((r, i) => (
                  <div
                    key={i}
                    className="rounded-2xl border border-slate-800 bg-[#0f172a] p-5 transition hover:border-purple-500/30"
                  >
                    <div className="font-black tracking-tight text-slate-200">
                      {r.k}
                    </div>
                    <p className="mt-2 text-sm leading-relaxed text-slate-400">
                      {r.v}
                    </p>
                  </div>
                )) ?? (
                  <div className="rounded-2xl border border-dashed border-slate-700 bg-[#0f172a] p-5 text-sm text-slate-500">
                    표시할 preview가 없습니다.
                  </div>
                )}
              </div>
            </div>
          </div>
        </section>
      )}

      {(spec.sections?.guard || spec.sections?.allowedVariation) && (
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-emerald-400">
            <ShieldCheck size={22} />
            <h2 className="text-2xl font-black tracking-tight text-white">
              허용 변화
            </h2>
          </div>

          {spec.sections?.guard?.cards?.length ? (
            <div className="grid grid-cols-1 gap-6 md:grid-cols-2 xl:grid-cols-3">
              {spec.sections.guard.cards.map((c) => (
                <div
                  key={c.id}
                  className="group min-w-0 rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8 shadow-xl transition hover:border-emerald-500/30"
                >
                  <div className="mb-6 flex items-center justify-between">
                    <IconBadge icon={c.icon} />
                    <span className="font-mono text-[10px] font-bold tracking-widest text-slate-600">
                      규칙 {c.id}
                    </span>
                  </div>

                  <h3 className="mb-3 text-xl font-black tracking-tighter text-white">
                    {c.title}
                  </h3>

                  <p className="mb-5 text-sm leading-relaxed text-slate-400">
                    {c.desc}
                  </p>

                  {c.metric && (
                    <div className="min-w-0 rounded-2xl border border-slate-800 bg-[#0f172a] p-4 font-mono text-[11px] text-purple-200/70">
                      <div className="overflow-x-auto scrollbar-hide">
                        <div className="w-max">
                          <InlineMath math={c.metric} />
                        </div>
                      </div>
                    </div>
                  )}

                  {c.note && (
                    <div className="mt-4 text-[11px] font-bold tracking-widest text-emerald-400">
                      {c.note}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="rounded-[2rem] border border-dashed border-slate-700 bg-[#1e293b] p-8 text-sm text-slate-500">
              허용 변화 카드가 아직 정의되지 않았습니다.
            </div>
          )}
        </section>
      )}

      {(spec.sections?.tolerance || spec.sections?.constraints) && (
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-rose-400">
            <Gauge size={22} />
            <h2 className="text-2xl font-black tracking-tight text-white">
              허용 한계
            </h2>
          </div>

          <p className="max-w-3xl text-sm leading-relaxed text-slate-400">
            exact equality가 항상 필요한 것은 아닙니다. 이 섹션은 downstream
            의미를 깨지 않는 실질적인 허용 한계를 정의합니다.
          </p>

          <div className="grid grid-cols-1 gap-6 md:grid-cols-2">
            {(spec.sections.tolerance?.cards ??
              spec.sections.constraints?.cards ??
              []
            ).map((card) => (
              <div
                key={card.id ?? card.title}
                className="rounded-[2.25rem] border border-slate-800 bg-[#1e293b] p-7 transition hover:border-rose-500/30"
              >
                <div className="mb-3 text-[10px] font-black tracking-[0.22em] text-slate-500">
                  Tolerance
                </div>
                <h3 className="text-lg font-black tracking-tight text-white">
                  {card.title}
                </h3>
                <p className="mt-3 text-sm leading-relaxed text-slate-400">
                  {card.desc}
                </p>
                {card.metric && (
                  <div className="mt-4 rounded-2xl border border-slate-800 bg-[#0f172a] p-4 font-mono text-[11px] text-rose-200/80">
                    <div className="overflow-x-auto scrollbar-hide">
                      <div className="w-max">
                        <InlineMath math={card.metric} />
                      </div>
                    </div>
                  </div>
                )}
                {card.note && (
                  <div className="mt-4 text-[11px] font-bold tracking-widest text-rose-300">
                    {card.note}
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
            <h2 className="text-2xl font-black tracking-tight text-white">
              반드시 보존되어야 하는 것
            </h2>
          </div>

          <div className="grid grid-cols-1 gap-5 md:grid-cols-2">
            {spec.sections.preserves.items?.map((item, i) => (
              <div
                key={i}
                className="rounded-[2rem] border border-slate-800 bg-[#1e293b] p-6 text-sm font-bold tracking-wide text-slate-200"
              >
                {item}
              </div>
            ))}
          </div>
        </section>
      )}

      {spec.sections?.failure && (
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-amber-400">
            <Scale size={22} />
            <h2 className="text-2xl font-black tracking-tight text-white">
              실패 경계
            </h2>
          </div>

          <p className="max-w-3xl text-sm leading-relaxed text-slate-400">
            아래 조건을 넘으면 단순한 numeric drift가 아니라 실제 semantic
            failure로 봐야 합니다.
          </p>

          <div className="space-y-4">
            {spec.sections.failure.items?.map((item, i) => (
              <div
                key={i}
                className="rounded-[2rem] border border-slate-800 bg-[#1e293b] p-6 text-sm leading-relaxed text-slate-300"
              >
                {item}
              </div>
            ))}
          </div>
        </section>
      )}

      {(spec.sections?.downstreamImpact ||
        spec.sections?.signals ||
        spec.sections?.applicableScenarios) && (
        <section className="space-y-8">
          <div className="flex items-center gap-3 text-cyan-400">
            <Target size={22} />
            <h2 className="text-2xl font-black tracking-tight text-white">
              후속 영향
            </h2>
          </div>

          <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
            {(spec.sections.downstreamImpact?.items ??
              spec.sections.signals?.bullets ??
              []
            ).length > 0 && (
              <div className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8">
                <div className="mb-4 text-[10px] font-black tracking-[0.22em] text-slate-500">
                  결정 안정성
                </div>
                <div className="space-y-4">
                  {(spec.sections.downstreamImpact?.items ??
                    spec.sections.signals?.bullets ??
                    []
                  ).map((item, i) => {
                    const titleText = item.k ?? item.title ?? `항목 ${i + 1}`;
                    const valueText = item.v ?? item.desc ?? item;

                    return (
                      <div
                        key={i}
                        className="rounded-2xl border border-slate-800 bg-[#0f172a] p-5"
                      >
                        <div className="font-black tracking-tight text-white">
                          {titleText}
                        </div>
                        <p className="mt-2 text-sm leading-relaxed text-slate-400">
                          {valueText}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {spec.sections.applicableScenarios?.items?.length > 0 && (
              <div className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8">
                <div className="mb-4 text-[10px] font-black tracking-[0.22em] text-slate-500">
                  적용 시나리오
                </div>
                <div className="space-y-3">
                  {spec.sections.applicableScenarios.items.map((item, i) => (
                    <div
                      key={i}
                      className="rounded-2xl border border-slate-800 bg-[#0f172a] p-4 text-sm font-bold text-slate-300"
                    >
                      {item}
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        </section>
      )}

      {spec.sections?.relatedConstructions && (
        <div className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8">
          <div className="mb-6 flex items-center gap-3 text-purple-400">
            <Boxes size={22} />
            <h2 className="text-2xl font-black tracking-tight text-white">
              관련 연산
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
            <h2 className="text-2xl font-black tracking-tight text-white">
              관련 realization
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

      <section className="space-y-8">
        <div className="flex items-center gap-3 text-purple-400">
          <Workflow size={22} />
          <h2 className="text-2xl font-black tracking-tight text-white">
            이 규칙과 연결되는 연산
          </h2>
        </div>

        <p className="max-w-3xl text-sm leading-relaxed text-slate-400">
          현재 선택된 보존 규칙 또는 허용 경계와 얼마나 강하게 연결되는지
          기준으로 정렬한 연산들입니다.
        </p>

        {matchedOps.length > 0 ? (
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2 xl:grid-cols-3">
            {matchedOps.map((match) => (
              <RuleMatchCard key={`${spec.id}-${match.opId}`} match={match} />
            ))}
          </div>
        ) : (
          <div className="rounded-[2rem] border border-dashed border-slate-700 bg-[#1e293b] p-8 text-sm text-slate-500">
            아직 이 규칙과 연결된 연산 프로파일이 없습니다.
          </div>
        )}
      </section>

      <div className="flex flex-col items-center justify-center gap-4 border-t border-slate-800 pt-10 sm:flex-row">
        <Link
          to="/compute/invariants"
          className="flex items-center gap-2 text-sm font-black text-purple-400 transition hover:text-white"
        >
          <BookOpen size={16} /> Preservation Guide로 돌아가기
        </Link>

        <Link
          to="/compute/ops"
          className="flex items-center gap-2 text-sm font-black text-emerald-400 transition hover:text-white"
        >
          <ArrowRight size={16} /> Ops Explorer 보기
        </Link>
      </div>
    </div>
  );
}

export default function PreservationGuidePage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const activeRuleId = searchParams.get("invariant");
  const isMain = !activeRuleId;

  const spec =
    activeRuleId && theoryByInvariantId[activeRuleId]
      ? theoryByInvariantId[activeRuleId]
      : null;

  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activeRuleId]);

  const matchedOps = useMemo(() => {
    if (!spec) return [];
    const profileKey = toRuleProfileKey(spec.id);

    return Object.entries(opsByInvariant ?? {})
      .map(([opId, invariantMap]) => {
        const entry = invariantMap?.[profileKey];
        if (!entry) return null;
        return { opId, ...entry };
      })
      .filter(Boolean)
      .sort((a, b) => (b.score ?? -1) - (a.score ?? -1));
  }, [spec]);

  const handleSelectRule = (id) => {
    setSearchParams({ invariant: id }, { replace: true });
    setIsSidebarOpen(false);
  };

  if (activeRuleId && !spec) {
    return (
      <div className="flex min-h-screen flex-col items-center justify-center bg-[#0f172a] p-10 font-mono text-blue-400">
        <div className="mb-4 animate-pulse text-2xl font-black uppercase">
          Analyzing...
        </div>
        <div className="text-sm text-slate-500">
          해당 rule 명세가 존재하지 않습니다.
        </div>
        <Link
          to="/compute/invariants"
          className="mt-6 rounded-xl bg-blue-600 px-4 py-2 font-bold text-white"
        >
          Back to Preservation Guide
        </Link>
      </div>
    );
  }

  return (
    <div className="flex min-h-dvh overflow-x-hidden bg-[#0f172a] text-slate-200 antialiased">
      <div className="lg:hidden">
        <button
          type="button"
          onClick={() => setIsSidebarOpen(true)}
          className="fixed left-4 top-4 z-50 rounded-2xl border border-slate-800 bg-[#0f172a] p-3 text-slate-200 shadow-xl backdrop-blur"
          aria-label="Open sidebar"
        >
          <Menu size={20} />
        </button>
      </div>

      <ComputeSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeInvariantId={activeRuleId || ""}
        quickInvariants={quickRules}
        version="v1.2.0 Preservation View"
      />

      <main className="flex min-w-0 flex-1 flex-col">
        <div className="flex-1 space-y-12 overflow-y-auto px-6 pb-10 pt-6 sm:px-8 sm:pb-12 sm:pt-8">
          {isMain ? (
            <PreservationGuideView
              groups={theoryInvariantGroups}
              onSelectRule={handleSelectRule}
            />
          ) : (
            <PreservationDetailView spec={spec} matchedOps={matchedOps} />
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