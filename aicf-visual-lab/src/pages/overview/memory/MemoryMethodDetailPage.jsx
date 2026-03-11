import React from "react";
import { useParams, Navigate, Link } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";
import {
  memoryMethodsMap,
  memoryPhaseLabels,
} from "../../../data/memory/methods";

export default function MemoryMethodDetailPage() {
  const { methodId, phaseId } = useParams();
  const method = memoryMethodsMap[methodId];

  if (!method) {
    return <Navigate to="/memory/methods" replace />;
  }

  const activePhase = phaseId || "overview";
  const validPhase =
    activePhase === "overview" || method.phases.includes(activePhase);

  if (!validPhase) {
    return <Navigate to={`/memory/methods/${methodId}`} replace />;
  }

  return (
    <div className="flex-1 overflow-y-auto bg-[#0f172a] text-slate-200 p-6 md:p-12">
      <div className="max-w-5xl mx-auto">
        <Link
          to="/memory/methods"
          className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition mb-8"
        >
          <ArrowLeft size={16} />
          Back to Methods
        </Link>

        <div className="mb-8">
          <div className="mb-4">{method.icon}</div>
          <p className="text-xs uppercase tracking-[0.25em] text-emerald-400 font-black mb-3">
            {method.category}
          </p>
          <h1 className="text-4xl md:text-5xl font-black text-white mb-4">
            {method.label}
          </h1>
          <p className="text-slate-400 text-lg leading-relaxed max-w-3xl">
            {method.desc}
          </p>
        </div>

        <div className="flex flex-wrap gap-3 mb-10">
          <Link
            to={`/memory/methods/${methodId}`}
            className={`px-4 py-2 rounded-xl text-sm font-bold border transition ${
              !phaseId
                ? "bg-emerald-600/10 text-emerald-400 border-emerald-500/20"
                : "text-slate-400 border-slate-800 hover:text-white hover:bg-slate-800"
            }`}
          >
            Overview
          </Link>

          {method.phases.map((phase) => (
            <Link
              key={phase}
              to={`/memory/methods/${methodId}/${phase}`}
              className={`px-4 py-2 rounded-xl text-sm font-bold border transition ${
                phaseId === phase
                  ? "bg-emerald-600/10 text-emerald-400 border-emerald-500/20"
                  : "text-slate-400 border-slate-800 hover:text-white hover:bg-slate-800"
              }`}
            >
              {memoryPhaseLabels[phase]}
            </Link>
          ))}
        </div>

        {!phaseId && (
          <section className="rounded-[2rem] border border-slate-800 bg-[#111827] p-8">
            <h2 className="text-2xl font-black text-white mb-4">
              Method Overview
            </h2>
            <p className="text-slate-400 leading-relaxed mb-6">
              여기에는 이 기법이 어떤 문제를 해결하는지, 어떤 수학적 성질을 활용하는지,
              왜 memory-centric optimization에서 중요한지를 요약해서 넣으면 됩니다.
            </p>

            <div className="flex flex-wrap gap-2">
              {method.tags.map((tag) => (
                <span
                  key={tag}
                  className="px-3 py-1 rounded-full bg-slate-900 text-[10px] font-bold text-slate-500 border border-slate-800"
                >
                  #{tag}
                </span>
              ))}
            </div>
          </section>
        )}

        {phaseId === "theory" && (
          <section className="rounded-[2rem] border border-slate-800 bg-[#111827] p-8">
            <h2 className="text-2xl font-black text-white mb-4">
              {memoryPhaseLabels.theory}
            </h2>
            <p className="text-slate-400 leading-relaxed">
              수학적 결합 가능성, online reduction 조건, normalization invariant,
              weighted accumulation reformulation 같은 내용을 여기에 넣으면 됩니다.
            </p>
          </section>
        )}

        {phaseId === "hardware" && (
          <section className="rounded-[2rem] border border-slate-800 bg-[#111827] p-8">
            <h2 className="text-2xl font-black text-white mb-4">
              {memoryPhaseLabels.hardware}
            </h2>
            <p className="text-slate-400 leading-relaxed">
              SRAM/L1/shared memory residency, HBM write-back 회피, tile size,
              bandwidth bottleneck, occupancy trade-off 같은 내용을 넣으면 됩니다.
            </p>
          </section>
        )}

        {phaseId === "compiler" && (
          <section className="rounded-[2rem] border border-slate-800 bg-[#111827] p-8">
            <h2 className="text-2xl font-black text-white mb-4">
              {memoryPhaseLabels.compiler}
            </h2>
            <p className="text-slate-400 leading-relaxed">
              MCIR에서 이 기법을 어떤 property로 표현할지, legality check,
              lowering rule, schedule rewrite, kernel mapping 조건을 여기에 넣으면 됩니다.
            </p>
          </section>
        )}

        <div className="mt-10">
          <Link
            to={phaseId ? `/memory/methods/${methodId}` : `/memory/methods/${methodId}/theory`}
            className="inline-flex items-center gap-2 text-emerald-400 font-bold hover:text-emerald-300 transition"
          >
            {phaseId ? "Go to method overview" : "Start with Math & Logic"}
            <ArrowRight size={16} />
          </Link>
        </div>
      </div>
    </div>
  );
}