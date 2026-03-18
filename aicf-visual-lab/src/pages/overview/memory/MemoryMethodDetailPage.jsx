import React, { useMemo, useState } from "react";
import { useParams, Navigate, Link } from "react-router-dom";
import {
  ArrowLeft,
  Menu,
  HardDrive,
  Database,
  ChevronRight,
} from "lucide-react";

import MemorySidebar from "../../../components/layout/MemorySidebar.jsx";
import { memoryMethodCatalogMap } from "../../../data/memory/methodCatalog";
import { memoryMethodDetails } from "../../../data/memory/methodDetails";

const sectionOrder = ["overview", "theory", "hardware", "compiler"];

const sectionMeta = {
  overview: {
    label: "Overview",
    anchor: "overview",
  },
  theory: {
    label: "Math & Logic",
    anchor: "math-logic",
  },
  hardware: {
    label: "Physical Analysis",
    anchor: "physical-analysis",
  },
  compiler: {
    label: "MCIR Implementation",
    anchor: "mcir-implementation",
  },
};

function SectionBlock({ sectionKey, content }) {
  const meta = sectionMeta[sectionKey];
  if (!content) return null;

  const hasOverviewFields =
    content.summary || content.problem || content.property || content.impact;

  const hasBody = Array.isArray(content.body) && content.body.length > 0;
  const hasBullets = Array.isArray(content.bullets) && content.bullets.length > 0;

  return (
    <section
      id={meta.anchor}
      className="scroll-mt-28 rounded-[2rem] border border-slate-800 bg-[#1e293b]/50 p-8 md:p-10"
    >
      <div className="mb-8">
        <div className="text-xs font-black uppercase tracking-[0.25em] text-emerald-400 mb-3">
          {meta.label}
        </div>
        <h2 className="text-2xl md:text-3xl font-black text-white tracking-tight">
          {content.title || meta.label}
        </h2>
      </div>

      {hasOverviewFields && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {content.summary && (
            <div className="rounded-[1.5rem] border border-slate-800 bg-slate-900/40 p-6">
              <div className="text-xs font-black uppercase tracking-[0.2em] text-emerald-400 mb-3">
                Summary
              </div>
              <p className="text-sm leading-relaxed text-slate-300">
                {content.summary}
              </p>
            </div>
          )}

          {content.problem && (
            <div className="rounded-[1.5rem] border border-slate-800 bg-slate-900/40 p-6">
              <div className="text-xs font-black uppercase tracking-[0.2em] text-emerald-400 mb-3">
                Problem
              </div>
              <p className="text-sm leading-relaxed text-slate-300">
                {content.problem}
              </p>
            </div>
          )}

          {content.property && (
            <div className="rounded-[1.5rem] border border-slate-800 bg-slate-900/40 p-6">
              <div className="text-xs font-black uppercase tracking-[0.2em] text-emerald-400 mb-3">
                Core Property
              </div>
              <p className="text-sm leading-relaxed text-slate-300">
                {content.property}
              </p>
            </div>
          )}

          {content.impact && (
            <div className="rounded-[1.5rem] border border-slate-800 bg-slate-900/40 p-6">
              <div className="text-xs font-black uppercase tracking-[0.2em] text-emerald-400 mb-3">
                Impact
              </div>
              <p className="text-sm leading-relaxed text-slate-300">
                {content.impact}
              </p>
            </div>
          )}
        </div>
      )}

      {(hasOverviewFields && (hasBody || hasBullets)) && (
        <div className="h-px bg-slate-800 my-8" />
      )}

      {(hasBody || hasBullets) && (
        <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_280px] gap-8">
          <div className="space-y-5">
            {hasBody &&
              content.body.map((paragraph, idx) => (
                <p
                  key={`${sectionKey}-body-${idx}`}
                  className="text-sm md:text-base leading-relaxed text-slate-300"
                >
                  {paragraph}
                </p>
              ))}
          </div>

          {hasBullets && (
            <aside className="rounded-[1.5rem] border border-slate-800 bg-slate-900/40 p-6 h-fit">
              <div className="text-xs font-black uppercase tracking-[0.2em] text-emerald-400 mb-4">
                Key Points
              </div>
              <div className="space-y-3">
                {content.bullets.map((bullet, idx) => (
                  <div
                    key={`${sectionKey}-bullet-${idx}`}
                    className="flex items-start gap-3 text-sm text-slate-300"
                  >
                    <ChevronRight
                      size={16}
                      className="mt-[2px] shrink-0 text-emerald-400"
                    />
                    <span>{bullet}</span>
                  </div>
                ))}
              </div>
            </aside>
          )}
        </div>
      )}
    </section>
  );
}

export default function MemoryMethodDetailPage() {
  const { methodId } = useParams();
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  const method = memoryMethodCatalogMap[methodId];
  const detail = memoryMethodDetails[methodId];

  const sections = useMemo(() => {
    if (!detail) return [];
    return sectionOrder
      .map((key) => ({
        key,
        content: detail[key],
        meta: sectionMeta[key],
      }))
      .filter((section) => section.content);
  }, [detail]);

  if (!method || !detail) {
    return <Navigate to="/memory/methods" replace />;
  }

  const Icon = method.icon;

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
          <div className="max-w-6xl mx-auto">
            <Link
              to="/memory/methods"
              className="inline-flex items-center gap-2 text-slate-400 hover:text-white transition mb-8 group"
            >
              <ArrowLeft
                size={16}
                className="group-hover:-translate-x-1 transition-transform"
              />
              <span className="text-sm font-bold uppercase tracking-wider">
                Back to Methods Library
              </span>
            </Link>

            <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,1fr)_240px] gap-10">
              <div className="space-y-10">
                {/* Hero */}
                <section className="rounded-[2.5rem] border border-slate-800 bg-[#1e293b] p-8 md:p-12 shadow-2xl relative overflow-hidden">
                  <div className="absolute -top-8 -right-8 text-[100px] font-black text-emerald-500/5 pointer-events-none">
                    MCIR
                  </div>

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

                  <p className="text-slate-400 text-lg leading-relaxed max-w-3xl font-light mb-8">
                    {method.desc}
                  </p>

                  <div className="flex flex-wrap gap-2">
                    {method.tags.map((tag) => (
                      <span
                        key={tag}
                        className="px-3 py-1 rounded-full bg-slate-900 text-[10px] font-bold text-slate-400 border border-slate-800"
                      >
                        #{tag}
                      </span>
                    ))}
                  </div>
                </section>

                {/* Sections */}
                {sections.map((section) => (
                  <SectionBlock
                    key={section.key}
                    sectionKey={section.key}
                    content={section.content}
                  />
                ))}
              </div>

              {/* Section Nav */}
              <aside className="hidden xl:block">
                <div className="sticky top-10 rounded-[1.75rem] border border-slate-800 bg-[#1e293b]/60 p-6">
                  <div className="text-xs font-black uppercase tracking-[0.25em] text-emerald-400 mb-5">
                    On This Page
                  </div>

                  <div className="space-y-3">
                    {sections.map((section) => (
                      <a
                        key={section.key}
                        href={`#${section.meta.anchor}`}
                        className="block text-sm text-slate-400 hover:text-white transition"
                      >
                        {section.meta.label}
                      </a>
                    ))}
                  </div>
                </div>
              </aside>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}