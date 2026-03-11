import React, { useState } from "react";
import { useParams, Navigate, Link } from "react-router-dom";
import { ArrowLeft, Menu, HardDrive, Database } from "lucide-react";

import MemorySidebar from "../../../components/layout/MemorySidebar.jsx";
import MethodPhaseTabs from "../../../features/memory/MethodPhaseTabs.jsx";
import MethodContentCard from "../../../features/memory/MethodContentCard.jsx";
import MethodPhasePager from "../../../features/memory/MethodPhasePager.jsx";

import { memoryMethodCatalogMap } from "../../../data/memory/methodCatalog";
import { memoryMethodDetails } from "../../../data/memory/methodDetails";
import { memoryPhaseLabels } from "../../../data/memory/phaseLabels";

const emptyContent = (phase) => ({
  title: memoryPhaseLabels[phase] || "Content",
  summary: "",
  problem: "",
  property: "",
  impact: "",
  body: [],
  bullets: [],
});

export default function MemoryMethodDetailPage() {
  const { methodId, phaseId } = useParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const method = memoryMethodCatalogMap[methodId];
  const detail = memoryMethodDetails[methodId];

  if (!method || !detail) {
    return <Navigate to="/memory/methods" replace />;
  }

  const activePhase = phaseId || "overview";
  const phaseSequence = ["overview", ...method.phases];

  if (!phaseSequence.includes(activePhase)) {
    return <Navigate to={`/memory/methods/${methodId}`} replace />;
  }

  const Icon = method.icon;
  const content = detail[activePhase] || emptyContent(activePhase);

  const currentIndex = phaseSequence.indexOf(activePhase);
  const prevPhase = phaseSequence[currentIndex - 1];
  const nextPhase = phaseSequence[currentIndex + 1];

  const getPhaseLink = (phase) =>
    phase === "overview"
      ? `/memory/methods/${methodId}`
      : `/memory/methods/${methodId}/${phase}`;

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <MemorySidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.0.6 Lab-Ready"
      />

      <main className="flex-1 flex flex-col min-w-0 font-sans">
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-5 py-4">
            <Link to="/memory/methods" className="flex items-center gap-2">
              <div className="bg-emerald-600 p-2 rounded-xl">
                <HardDrive size={18} className="text-white" />
              </div>
              <div className="font-black text-emerald-400 tracking-tight">
                AICF MEMORY
              </div>
            </Link>

            <button
              type="button"
              aria-label="Open sidebar"
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200"
            >
              <Menu size={18} />
            </button>
          </div>
        </header>

        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 md:p-12 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          <div className="max-w-5xl mx-auto">
            <Link
              to="/memory/methods"
              className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition mb-8 group"
            >
              <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
              <span className="text-sm font-bold uppercase tracking-wider">
                Back to Methods Library
              </span>
            </Link>

            <div className="mb-12">
              <div className="mb-6 inline-flex p-5 rounded-[1.5rem] bg-slate-900/60 border border-slate-800 shadow-xl shadow-emerald-900/10">
                <Icon className={method.iconColor} size={40} />
              </div>

              <div className="flex items-center gap-2 text-emerald-400 font-mono text-[10px] font-black uppercase tracking-[0.3em] mb-4">
                <Database size={14} />
                {method.category}
              </div>

              <h1 className="text-4xl md:text-6xl font-black text-white mb-6 tracking-tight">
                {method.label}
              </h1>

              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl font-light">
                {method.desc}
              </p>
            </div>

            <MethodPhaseTabs
              phaseSequence={phaseSequence}
              activePhase={activePhase}
              getPhaseLink={getPhaseLink}
            />

            <MethodContentCard content={content} Icon={Icon} />

            <MethodPhasePager
              prevPhase={prevPhase}
              nextPhase={nextPhase}
              getPhaseLink={getPhaseLink}
            />
          </div>
        </div>
      </main>
    </div>
  );
}