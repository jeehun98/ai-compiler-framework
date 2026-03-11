import React, { useState } from "react";
import { useParams, Navigate, Link } from "react-router-dom";
import { ArrowLeft, ArrowRight, Menu, HardDrive, Database } from "lucide-react";
import {
  memoryMethodsMap,
  memoryPhaseLabels,
} from "../../../data/memory/methods";
import { memoryMethodDetails } from "../../../data/memory/methods/index";
import MemorySidebar from "../../../components/MemorySidebar.jsx";

export default function MemoryMethodDetailPage() {
  const { methodId, phaseId } = useParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const method = memoryMethodsMap[methodId];
  const detail = memoryMethodDetails[methodId];

  if (!method || !detail) {
    return <Navigate to="/memory/methods" replace />;
  }

  const activePhase = phaseId || "overview";
  const phaseSequence = ["overview", ...method.phases];
  const validPhase = phaseSequence.includes(activePhase);

  if (!validPhase) {
    return <Navigate to={`/memory/methods/${methodId}`} replace />;
  }

  const Icon = method.icon;
  const content = detail[activePhase];
  const currentIndex = phaseSequence.indexOf(activePhase);
  const prevPhase = phaseSequence[currentIndex - 1];
  const nextPhase = phaseSequence[currentIndex + 1];

  const getPhaseLink = (phase) =>
    phase === "overview"
      ? `/memory/methods/${methodId}`
      : `/memory/methods/${methodId}/${phase}`;

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      {/* GLOBAL SIDEBAR - 일관성 유지 */}
      <MemorySidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.0.6 Lab-Ready"
      />

      <main className="flex-1 flex flex-col min-w-0 font-sans">
        {/* Mobile Header */}
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-5 py-4">
            <Link to="/memory/methods" className="flex items-center gap-2">
              <div className="bg-emerald-600 p-2 rounded-xl">
                <HardDrive size={18} className="text-white" />
              </div>
              <div className="font-black text-emerald-400 tracking-tight">AICF MEMORY</div>
            </Link>
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200"
            >
              <Menu size={18} />
            </button>
          </div>
        </header>

        {/* Mobile Spacer */}
        <div className="md:hidden h-[68px]" />

        {/* Content Area */}
        <div className="flex-1 overflow-y-auto p-6 md:p-12 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          <div className="max-w-5xl mx-auto">
            {/* Navigation Breadcrumb */}
            <Link
              to="/memory/methods"
              className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition mb-8 group"
            >
              <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
              <span className="text-sm font-bold uppercase tracking-wider">Back to Methods Library</span>
            </Link>

            {/* Header Section */}
            <div className="mb-12">
              <div className="mb-6 inline-flex p-5 rounded-[1.5rem] bg-slate-900/60 border border-slate-800 shadow-xl shadow-emerald-900/10">
                <Icon className={method.iconColor} size={40} />
              </div>

              <div className="flex items-center gap-2 text-emerald-400 font-mono text-[10px] font-black uppercase tracking-[0.3em] mb-4">
                <Database size={14} /> {method.category}
              </div>

              <h1 className="text-4xl md:text-6xl font-black text-white mb-6 tracking-tight">
                {method.label}
              </h1>

              <p className="text-slate-400 text-lg leading-relaxed max-w-3xl font-light">
                {method.desc}
              </p>
            </div>

            {/* Phase Navigation Tabs */}
            <div className="flex flex-wrap gap-2 mb-10 p-1 bg-slate-900/40 w-fit rounded-2xl border border-slate-800/50">
              {phaseSequence.map((phase) => (
                <Link
                  key={phase}
                  to={getPhaseLink(phase)}
                  className={`px-5 py-2.5 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${
                    activePhase === phase
                      ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20"
                      : "text-slate-500 hover:text-slate-200 hover:bg-slate-800"
                  }`}
                >
                  {memoryPhaseLabels[phase]}
                </Link>
              ))}
            </div>

            {/* Main Spec Card */}
            <section className="rounded-[2.5rem] border border-slate-800 bg-[#111827] p-8 md:p-12 space-y-10 shadow-2xl relative overflow-hidden">
                <div className="absolute top-0 right-0 p-12 opacity-[0.02] pointer-events-none">
                    <Icon size={300} />
                </div>
                
              <div className="relative z-10">
                <h2 className="text-2xl md:text-3xl font-black text-white mb-6 flex items-center gap-3">
                  <span className="w-8 h-[2px] bg-emerald-500"></span>
                  {content.title}
                </h2>

                {content.summary && (
                  <p className="text-slate-300 text-lg leading-relaxed font-light mb-10 italic border-l-4 border-emerald-500/30 pl-6">
                    {content.summary}
                  </p>
                )}

                <div className="grid grid-cols-1 gap-10">
                    {content.problem && (
                        <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800">
                            <h3 className="text-emerald-400 text-[10px] font-black uppercase tracking-widest mb-3">The Problem</h3>
                            <p className="text-slate-400 leading-relaxed">{content.problem}</p>
                        </div>
                    )}

                    <div className="space-y-8">
                        {content.property && (
                            <div>
                                <h3 className="text-white text-lg font-black mb-3">Key Mechanism</h3>
                                <p className="text-slate-400 leading-relaxed">{content.property}</p>
                            </div>
                        )}

                        {content.impact && (
                            <div>
                                <h3 className="text-white text-lg font-black mb-3">Architectural Impact</h3>
                                <p className="text-slate-400 leading-relaxed">{content.impact}</p>
                            </div>
                        )}

                        {content.body && (
                            <div className="space-y-4 pt-4 border-t border-slate-800">
                                {content.body.map((paragraph, idx) => (
                                    <p key={idx} className="text-slate-400 leading-relaxed font-light">
                                        {paragraph}
                                    </p>
                                ))}
                            </div>
                        )}
                    </div>
                </div>

                {content.bullets && (
                  <div className="flex flex-wrap gap-2 mt-10">
                    {content.bullets.map((item) => (
                      <span
                        key={item}
                        className="px-4 py-1.5 rounded-full bg-emerald-500/5 text-[10px] font-black text-emerald-400/80 border border-emerald-500/10"
                      >
                        {item}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </section>

            {/* Bottom Pagination */}
            <div className="mt-12 flex items-center justify-between border-t border-slate-800 pt-8">
              <div>
                {prevPhase && (
                  <Link
                    to={getPhaseLink(prevPhase)}
                    className="group flex flex-col items-start gap-1"
                  >
                    <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Previous Phase</span>
                    <div className="inline-flex items-center gap-2 text-slate-300 font-bold group-hover:text-white transition">
                        <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
                        {memoryPhaseLabels[prevPhase]}
                    </div>
                  </Link>
                )}
              </div>

              <div className="text-right">
                {nextPhase && (
                  <Link
                    to={getPhaseLink(nextPhase)}
                    className="group flex flex-col items-end gap-1"
                  >
                    <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">Next Phase</span>
                    <div className="inline-flex items-center gap-2 text-emerald-400 font-bold group-hover:text-emerald-300 transition">
                        {memoryPhaseLabels[nextPhase]}
                        <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
                    </div>
                  </Link>
                )}
              </div>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}