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
  Menu,
  ArrowUpRight,
  GitMerge,
  Search,
  BookOpen,
  Boxes,
  Layers,
} from "lucide-react";

import { useSearchParams, Link } from "react-router-dom";
import {
  allOpsData,
  opFamilyTraits,
} from "../../../data/ops/index.js";
import KernelDeepDive from "../../../components/common/KernelDeepDive.jsx";
import ComputeSidebar from "../../../components/layout/ComputeSidebar.jsx";

const PROPERTY_TONE = {
  strong: {
    badge: "bg-emerald-500/10 text-emerald-300 border-emerald-500/20",
    bar: "bg-emerald-400",
    label: "STRONG",
  },
  medium: {
    badge: "bg-blue-500/10 text-blue-300 border-blue-500/20",
    bar: "bg-blue-400",
    label: "MEDIUM",
  },
  conditional: {
    badge: "bg-amber-500/10 text-amber-300 border-amber-500/20",
    bar: "bg-amber-400",
    label: "CONDITIONAL",
  },
  limited: {
    badge: "bg-purple-500/10 text-purple-300 border-purple-500/20",
    bar: "bg-purple-400",
    label: "LIMITED",
  },
  weak: {
    badge: "bg-slate-500/10 text-slate-300 border-slate-500/20",
    bar: "bg-slate-400",
    label: "WEAK",
  },
  not_applicable: {
    badge: "bg-rose-500/10 text-rose-300 border-rose-500/20",
    bar: "bg-rose-400",
    label: "N/A",
  },
};

const FAMILY_TONE = {
  normalizationFamily: {
    title: "Normalization Family",
    accent: "text-purple-400",
    border: "hover:border-purple-500/30",
    chip: "border-purple-500/20 bg-purple-500/5 text-purple-300",
    keyText: "text-purple-300",
  },
  gatingFamily: {
    title: "Gating Family",
    accent: "text-amber-400",
    border: "hover:border-amber-500/30",
    chip: "border-amber-500/20 bg-amber-500/5 text-amber-300",
    keyText: "text-amber-300",
  },
  pathMergeFamily: {
    title: "Path Merge Family",
    accent: "text-emerald-400",
    border: "hover:border-emerald-500/30",
    chip: "border-emerald-500/20 bg-emerald-500/5 text-emerald-300",
    keyText: "text-emerald-300",
  },
  broadcastShiftFamily: {
    title: "Broadcast Shift Family",
    accent: "text-cyan-400",
    border: "hover:border-cyan-500/30",
    chip: "border-cyan-500/20 bg-cyan-500/5 text-cyan-300",
    keyText: "text-cyan-300",
  },
  linearProjectionFamily: {
    title: "Linear Projection Family",
    accent: "text-blue-400",
    border: "hover:border-blue-500/30",
    chip: "border-blue-500/20 bg-blue-500/5 text-blue-300",
    keyText: "text-blue-300",
  },
  competitionFamily: {
    title: "Competition Family",
    accent: "text-pink-400",
    border: "hover:border-pink-500/30",
    chip: "border-pink-500/20 bg-pink-500/5 text-pink-300",
    keyText: "text-pink-300",
  },
  stateUpdateFamily: {
    title: "State Update Family",
    accent: "text-orange-400",
    border: "hover:border-orange-500/30",
    chip: "border-orange-500/20 bg-orange-500/5 text-orange-300",
    keyText: "text-orange-300",
  },
  default: {
    title: "Family Trait",
    accent: "text-cyan-400",
    border: "hover:border-cyan-500/30",
    chip: "border-slate-500/20 bg-slate-500/5 text-slate-300",
    keyText: "text-slate-300",
  },
};

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
  const chosenSummary = data?.lowering?.chosen?.summary ?? "";
  const hasDeepDive = !!(data?.kernel_evolution || data?.evolution);

  const propertyEntries = Object.entries(data?.propertyProfile ?? {});
  const opConstraints = data?.opConstraints ?? [];
  const downstream = data?.downstreamConstraints ?? [];
  const loweringReasons = data?.lowering?.chosen?.reason ?? [];
  const costWeights = data?.costModel?.weights_hint?.default ?? {};

  const familyTraits = opFamilyTraits[activeOpId] ?? {};
  const familyEntries = Object.entries(familyTraits).filter(
    ([, value]) => value && typeof value === "object"
  );

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
        version="v1.1.0 Property View"
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
                  v1.1.0 Property View
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
                    Property-Guided Lowering
                  </span>
                </h1>

                <p className="mt-8 text-slate-400 text-lg sm:text-xl leading-relaxed max-w-3xl font-light">
                  Ops Explorer는 각 연산의 <strong>canonical spec</strong>과{" "}
                  <strong>property profile</strong>을 기준으로, 어떤 최적화가
                  허용되며 어떤 실행 형태로 이어질 수 있는지를 분석합니다.
                  <br />
                  즉, <strong>Property Atlas</strong>가 정의한 semantic
                  property를 개별 operator의 profile로 투영하고, 이를 실제
                  lowering candidate로 연결하는{" "}
                  <strong>property-to-lowering bridge</strong> 역할을 합니다.
                </p>

                <div className="mt-8 inline-flex items-center gap-2 rounded-2xl border border-blue-500/20 bg-blue-500/5 px-4 py-2 text-[11px] font-bold uppercase tracking-widest text-blue-300">
                  <ShieldCheck size={14} />
                  Semantic / Property View
                </div>
              </section>

              <section className="bg-[#0b1120] border border-slate-800 rounded-[2.5rem] p-8 sm:p-10">
                <div className="flex items-center gap-3 text-emerald-400 mb-6">
                  <GitMerge size={22} />
                  <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                    From Property Atlas to Realization
                  </h2>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                  {[
                    {
                      step: "01",
                      title: "Property Atlas",
                      desc: "재사용 가능한 semantic property와 rewrite law를 정의합니다.",
                    },
                    {
                      step: "02",
                      title: "Op Property Profile",
                      desc: "개별 연산이 어떤 property를 얼마나 만족하는지 판정합니다.",
                    },
                    {
                      step: "03",
                      title: "Lowering Family",
                      desc: "허용된 property 공간 안에서 자연스러운 realization family를 좁혀갑니다.",
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
                      title: "Op Canonical Spec",
                      desc: "수식, 축 의미, 입출력 관계를 통해 연산의 canonical form을 정리합니다.",
                      tag: "SPEC",
                    },
                    {
                      step: "2",
                      title: "Property Atlas Matching",
                      desc: "재사용 가능한 property 집합 중 어떤 성질이 이 연산에 성립하는지 평가합니다.",
                      tag: "PROPERTIES",
                    },
                    {
                      step: "3",
                      title: "Op-Specific Constraint 확인",
                      desc: "이 연산만의 상태 정렬성, 수치 안정성, 시간 순서 제약을 확인합니다.",
                      tag: "CONSTRAINTS",
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
                    </div>
                  ))}
                </div>
              </section>

              <section className="bg-blue-600/5 border border-blue-500/20 rounded-[3rem] p-12 text-center">
                <h2 className="text-2xl font-black text-white uppercase mb-4">
                  Select an Operator to Explore
                </h2>
                <p className="text-slate-400 text-sm max-w-2xl mx-auto leading-relaxed mb-8">
                  각 연산이 어떤 canonical spec을 가지며, 어떤 property profile과
                  op-specific constraint를 통해 어떤 lowering family로 이어질 수
                  있는지 확인할 수 있습니다.
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

              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 text-center">
                <div className="flex items-center justify-center gap-2 text-blue-400 mb-4">
                  <BookOpen size={18} />
                  <span className="text-[11px] font-black uppercase tracking-widest">
                    Property Layer
                  </span>
                </div>

                <h2 className="text-2xl font-black text-white uppercase mb-4">
                  Need the Property Atlas First?
                </h2>

                <p className="text-slate-400 max-w-2xl mx-auto leading-relaxed mb-8">
                  Ops Explorer는 개별 연산의 property profile과 lowering candidate를
                  설명합니다. 재사용 가능한 semantic property, rewrite law, 성립
                  조건과 제한 조건을 먼저 보려면 Property Atlas를 확인하세요.
                </p>

                <Link
                  to="/compute/properties"
                  className="inline-flex items-center gap-2 px-8 py-4 rounded-2xl bg-blue-600/10 border border-blue-500/20 text-blue-300 font-black uppercase tracking-widest hover:bg-blue-600/20 transition"
                >
                  View Property Atlas <ArrowUpRight size={16} />
                </Link>
              </section>
            </div>
          ) : (
            <div className="animate-in slide-in-from-bottom-4 duration-500 space-y-12">
              <section className="flex flex-col lg:flex-row lg:items-end justify-between gap-6 border-b border-slate-800 pb-8">
                <div className="space-y-2 min-w-0">
                  <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black uppercase tracking-[0.3em]">
                    <Activity size={14} /> Operator Property Report
                  </div>
                  <h2 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight break-words">
                    {data.id}{" "}
                    <span className="text-blue-500/30 font-light ml-2">
                      Explorer
                    </span>
                  </h2>
                </div>

                <div className="flex items-center gap-2 text-emerald-400 font-bold bg-emerald-400/5 px-4 py-2 rounded-xl border border-emerald-400/10 text-[11px] uppercase tracking-widest w-fit">
                  <ShieldCheck size={16} /> Property Anchored
                </div>
              </section>

              <section className="space-y-6">
                <div className="flex items-center gap-3 text-blue-400">
                  <Share2 size={24} />
                  <h3 className="text-2xl font-black uppercase tracking-tight">
                    1. Canonical Spec & Property Profile
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
                      Canonical dataflow & axis meaning
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
                            {semantic.axes[axis].description ||
                              interpretation?.[axis] ||
                              "정의되지 않음"}
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
                        Property Profile
                      </p>
                      <span className="text-[9px] text-emerald-500 font-bold bg-emerald-500/10 px-2 py-0.5 rounded border border-emerald-500/20">
                        ANALYZED
                      </span>
                    </div>

                    {propertyEntries.length > 0 ? (
                      propertyEntries.map(([key, prop]) => (
                        <PropertyCard key={key} propertyKey={key} prop={prop} />
                      ))
                    ) : (
                      <EmptyCard message="정의된 property profile이 없습니다." />
                    )}
                  </div>
                </div>
              </section>

              <section className="space-y-6">
                <div className="flex items-center gap-3 text-cyan-400">
                  <Boxes size={24} />
                  <h3 className="text-2xl font-black uppercase tracking-tight">
                    2. Family Traits
                  </h3>
                </div>

                <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
                  이 섹션은 property profile과 별개로, 해당 연산이 어떤 구조적
                  family에 속하는지를 보여줍니다. normalization, gating,
                  path-merge, projection, competition, state-update 같은
                  도메인 특화 패턴을 compact하게 표시합니다.
                </p>

                {familyEntries.length > 0 ? (
                  <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    {familyEntries.map(([familyKey, familyValue]) => (
                      <FamilyTraitCard
                        key={familyKey}
                        familyKey={familyKey}
                        familyValue={familyValue}
                      />
                    ))}
                  </div>
                ) : (
                  <EmptyCard message="정의된 family traits가 없습니다." />
                )}
              </section>

              <section className="space-y-6">
                <div className="flex items-center gap-3 text-purple-400">
                  <Layers size={24} />
                  <h3 className="text-2xl font-black uppercase tracking-tight">
                    3. Op-Specific Constraints
                  </h3>
                </div>

                <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
                  Property Atlas의 reusable property와 별개로, 이 연산만이
                  가지는 상태 정렬성, 수치 안정성, 시간 순서 제약을 표시합니다.
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                  {opConstraints.length > 0 ? (
                    opConstraints.map((constraint) => (
                      <div
                        key={constraint.id}
                        className="bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800 shadow-xl"
                      >
                        <div className="flex items-center justify-between gap-3 mb-4">
                          <div className="text-[10px] font-black uppercase tracking-widest text-purple-400">
                            {constraint.id}
                          </div>
                          <div className="text-[9px] text-slate-500 border border-slate-700 rounded px-2 py-0.5">
                            OP LOCAL
                          </div>
                        </div>

                        <h4 className="text-white font-black text-lg tracking-tight mb-3">
                          {constraint.name}
                        </h4>

                        {constraint.metric ? (
                          <div className="bg-[#0f172a] p-3 rounded-xl border border-slate-800 font-mono text-[11px] text-blue-200/70 mb-4 overflow-x-auto scrollbar-hide">
                            <InlineMath math={constraint.metric} />
                          </div>
                        ) : null}

                        <p className="text-sm text-slate-400 leading-relaxed mb-4">
                          {constraint.detail}
                        </p>

                        <div className="flex flex-wrap gap-2">
                          {(constraint.consequence ?? []).map((item) => (
                            <span
                              key={item}
                              className="text-[9px] font-bold bg-purple-500/10 text-purple-300 px-2 py-1 rounded border border-purple-500/20 uppercase tracking-tight"
                            >
                              {item}
                            </span>
                          ))}
                        </div>
                      </div>
                    ))
                  ) : (
                    <EmptyCard
                      className="lg:col-span-3"
                      message="정의된 op-specific constraint가 없습니다."
                    />
                  )}
                </div>
              </section>

              <section className="space-y-6">
                <div className="flex items-center gap-3 text-purple-400">
                  <Eye size={24} />
                  <h3 className="text-2xl font-black uppercase tracking-tight">
                    4. Constraint-Aware Lowering Strategy
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

              <section className="space-y-6">
                <div className="flex items-center gap-3 text-emerald-400">
                  <Zap size={24} />
                  <h3 className="text-2xl font-black uppercase tracking-tight">
                    5. Lowering Decision Snapshot
                  </h3>
                </div>

                <p className="text-slate-500 text-sm leading-relaxed max-w-3xl">
                  {data.descriptions?.realization ??
                    "이 섹션은 실행 메커니즘 전체를 설명하지 않고, 현재 연산이 어떤 realization family로 이어지는지와 그 선택 근거를 compact하게 요약합니다."}
                </p>

                <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">
                  <div className="lg:col-span-7 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
                    <div className="flex items-center gap-2 mb-6 text-emerald-400 font-black text-[10px] uppercase tracking-widest">
                      <Terminal size={16} /> Chosen Realization
                    </div>

                    <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                      <CompactMetaCard
                        label="Variant"
                        value={chosenVariant}
                        tone="blue"
                      />
                      <CompactMetaCard
                        label="Property Count"
                        value={`${propertyEntries.length}`}
                        tone="emerald"
                      />
                      <CompactMetaCard
                        label="Op Constraints"
                        value={`${opConstraints.length}`}
                        tone="purple"
                      />
                    </div>

                    {chosenSummary ? (
                      <p className="text-sm text-slate-400 leading-relaxed mb-6">
                        {chosenSummary}
                      </p>
                    ) : null}

                    <p className="text-[10px] text-slate-500 uppercase font-black mb-3 tracking-widest">
                      Why this family
                    </p>

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

                  <div className="lg:col-span-5 bg-[#1e293b] p-6 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
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
                        커널 내부 scheduling, memory movement, physical metrics는
                        Ops 페이지의 본체가 아니라 Deep Dive 계층에서 다룹니다.
                      </p>
                    </div>
                  </div>
                </div>
              </section>

              <section className="space-y-5">
                <div className="flex items-center gap-3 text-blue-400">
                  <BookOpen size={22} />
                  <h3 className="text-2xl font-black uppercase tracking-tight">
                    Related Views
                  </h3>
                </div>

                <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
                  <Link
                    to="/compute/properties"
                    className="group bg-[#1e293b] border border-slate-800 rounded-[2rem] p-6 hover:border-blue-500/30 transition"
                  >
                    <div className="text-[10px] font-black uppercase tracking-widest text-blue-500 mb-2">
                      Property Layer
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <h4 className="text-lg font-black text-white uppercase mb-2">
                          View Property Atlas
                        </h4>
                        <p className="text-sm text-slate-400 leading-relaxed">
                          재사용 가능한 semantic property, rewrite law, 성립 조건과
                          제한 조건을 확인합니다.
                        </p>
                      </div>
                      <ArrowUpRight
                        size={18}
                        className="text-slate-600 group-hover:text-blue-400 transition shrink-0"
                      />
                    </div>
                  </Link>

                  <button
                    type="button"
                    onClick={() => hasDeepDive && setIsModalOpen(true)}
                    disabled={!hasDeepDive}
                    className={`text-left bg-[#1e293b] border rounded-[2rem] p-6 transition ${
                      hasDeepDive
                        ? "border-slate-800 hover:border-emerald-500/30"
                        : "border-slate-800 opacity-60 cursor-not-allowed"
                    }`}
                  >
                    <div className="text-[10px] font-black uppercase tracking-widest text-emerald-500 mb-2">
                      Mechanism Layer
                    </div>
                    <div className="flex items-center justify-between gap-4">
                      <div>
                        <h4 className="text-lg font-black text-white uppercase mb-2">
                          Kernel Deep Dive
                        </h4>
                        <p className="text-sm text-slate-400 leading-relaxed">
                          구현 세부, scheduling, kernel evolution, memory path를
                          별도 계층에서 확인합니다.
                        </p>
                      </div>
                      <ArrowUpRight
                        size={18}
                        className={`transition shrink-0 ${
                          hasDeepDive ? "text-slate-600" : "text-slate-700"
                        }`}
                      />
                    </div>
                  </button>
                </div>
              </section>

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

                <div className="text-[10px] text-slate-700 uppercase tracking-[0.2em] font-bold">
                  Property → Profile → Family → Constraint → Realization
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

function PropertyCard({ propertyKey, prop }) {
  const tone = PROPERTY_TONE[prop?.status] ?? PROPERTY_TONE.weak;
  const score =
    typeof prop?.score === "number"
      ? Math.max(0, Math.min(100, Math.round(prop.score * 100)))
      : null;

  return (
    <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 hover:border-blue-500/30 transition shadow-lg relative overflow-hidden">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="min-w-0">
          <div className="text-blue-400 font-black text-xs uppercase tracking-tight break-words">
            {propertyKey}
          </div>
          {prop?.summary ? (
            <div className="text-[11px] text-slate-500 mt-1 leading-relaxed">
              {prop.summary}
            </div>
          ) : null}
        </div>

        <span
          className={`shrink-0 text-[9px] font-black px-2 py-1 rounded-lg border uppercase tracking-widest ${tone.badge}`}
        >
          {tone.label}
        </span>
      </div>

      {prop?.reason ? (
        <div className="bg-[#0f172a] p-3 rounded-xl border border-slate-800 font-mono text-[11px] text-blue-200/70 mb-4 italic overflow-x-auto scrollbar-hide">
          <InlineMath math={prop.reason} />
        </div>
      ) : null}

      {score !== null ? (
        <div className="mb-4">
          <div className="flex items-center justify-between text-[9px] uppercase font-black tracking-widest text-slate-500 mb-2">
            <span>Affinity</span>
            <span>{score}</span>
          </div>
          <div className="h-2 rounded-full bg-[#0f172a] border border-slate-800 overflow-hidden">
            <div
              className={`h-full ${tone.bar}`}
              style={{ width: `${score}%` }}
            />
          </div>
        </div>
      ) : null}

      <div className="flex flex-wrap gap-1">
        {(prop?.allows ?? []).map((item) => (
          <span
            key={item}
            className="text-[8px] font-bold bg-slate-900 text-slate-400 px-2 py-0.5 rounded border border-slate-800 uppercase tracking-tighter"
          >
            + {item}
          </span>
        ))}
      </div>
    </div>
  );
}

function FamilyTraitCard({ familyKey, familyValue }) {
  const tone = FAMILY_TONE[familyKey] ?? FAMILY_TONE.default;
  const label = tone.title;

  return (
    <div
      className={`bg-[#1e293b] p-6 rounded-[2rem] border border-slate-800 shadow-xl transition ${tone.border}`}
    >
      <div className="flex items-center justify-between gap-3 mb-4">
        <div>
          <div
            className={`text-[10px] font-black uppercase tracking-widest mb-2 ${tone.accent}`}
          >
            Family Trait
          </div>
          <h4 className="text-xl font-black uppercase tracking-tight text-white break-words">
            {label}
          </h4>
        </div>

        <div
          className={`text-[9px] font-black uppercase tracking-widest border rounded px-2 py-1 ${tone.chip}`}
        >
          STRUCTURAL
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {Object.entries(familyValue).map(([k, v]) => (
          <div
            key={k}
            className="rounded-2xl border border-slate-800 bg-[#0f172a] p-4"
          >
            <div
              className={`text-[9px] font-black uppercase tracking-widest mb-2 break-words ${tone.keyText}`}
            >
              {formatFamilyKey(k)}
            </div>
            <div className="text-sm font-bold text-slate-200 break-words">
              {formatFamilyValue(v)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function formatFamilyKey(key) {
  return key
    .replace(/([A-Z])/g, " $1")
    .replace(/_/g, " ")
    .trim();
}

function formatFamilyValue(value) {
  if (Array.isArray(value)) {
    return value.join(", ");
  }

  if (typeof value === "boolean") {
    return value ? "Yes" : "No";
  }

  if (value === null || value === undefined) {
    return "N/A";
  }

  return String(value);
}

function CompactMetaCard({ label, value, tone = "blue" }) {
  const toneClass = {
    blue: "text-blue-300 bg-blue-500/5 border-blue-500/10",
    emerald: "text-emerald-300 bg-emerald-500/5 border-emerald-500/10",
    purple: "text-purple-300 bg-purple-500/5 border-purple-500/10",
  }[tone];

  return (
    <div className={`rounded-2xl border p-4 ${toneClass}`}>
      <div className="text-[9px] font-black uppercase tracking-widest mb-2 opacity-70">
        {label}
      </div>
      <div className="text-sm font-black break-words">{value}</div>
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