// src/pages/TheoryPage.jsx
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
} from "lucide-react";

import AppSidebar from "../components/AppSidebar.jsx";
import { theoryByOpId, theoryOpIds } from "../data/theory/index.js";

const iconMap = {
  target: Target,
  binary: Binary,
  arrow: ArrowRightLeft,
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
  const [searchParams, setSearchParams] = useSearchParams();
  const initialOp = searchParams.get("op") || theoryOpIds?.[0] || "GEMM";

  const [activeOpId, setActiveOpId] = useState(initialOp);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    const opFromUrl = searchParams.get("op");
    if (opFromUrl && opFromUrl !== activeOpId) setActiveOpId(opFromUrl);
  }, [searchParams]);

  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activeOpId]);

  const spec = theoryByOpId[activeOpId];

  const onSelect = (id) => {
    setActiveOpId(id);
    setSearchParams({ op: id }, { replace: true });
  };

  // URL에 op 없으면 기본값 박아줌 (첫 진입 안정화)
  useEffect(() => {
    if (!searchParams.get("op")) {
      setSearchParams({ op: initialOp }, { replace: true });
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  if (!spec) {
    return (
      <div className="p-10 text-blue-400 bg-[#0f172a] min-h-screen flex flex-col items-center justify-center font-mono">
        <div className="animate-pulse mb-4 text-2xl font-black uppercase">Theory Not Found</div>
        <div className="text-slate-500 text-sm">
          data/theory에 "{activeOpId}" 스펙이 없습니다.
        </div>
        <div className="mt-6 flex flex-wrap gap-2 justify-center">
          {theoryOpIds.map((id) => (
            <button
              key={id}
              onClick={() => onSelect(id)}
              className="px-4 py-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200 text-xs font-bold"
            >
              {id}
            </button>
          ))}
        </div>
      </div>
    );
  }

  const { title, subtitle, hero, sections } = spec;
  const projection = sections?.projection;
  const equivalence = sections?.equivalence;
  const cost = sections?.cost;

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      {/* Mobile Header */}
      <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
        <div className="flex items-center justify-between px-5 py-4">
          <Link to="/" className="flex items-center gap-2">
            <div className="bg-blue-600 p-2 rounded-xl">
              <Cpu size={18} className="text-white" />
            </div>
            <div className="leading-none">
              <div className="font-black text-blue-400 tracking-tight">AICF LAB</div>
              <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">v1.0.4 Stable</div>
            </div>
          </Link>

          <button
            onClick={() => setIsSidebarOpen(true)}
            className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200"
            aria-label="Open sidebar"
          >
            <Menu size={18} />
          </button>
        </div>
      </header>

      <AppSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeOpId={activeOpId}
      />

      <main className="flex-1 flex flex-col min-w-0">
        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-14 pb-32 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          {/* HERO */}
          <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-12 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[140px] sm:text-[180px] font-black text-blue-500/5 pointer-events-none tracking-tighter">
              {spec.id}
            </div>

            <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black uppercase tracking-[0.35em]">
              <Waypoints size={14} /> {subtitle || "Theory Spec"}
            </div>

            <h1 className="mt-4 text-4xl sm:text-6xl font-black tracking-tight text-white leading-[1.05]">
              {title}
            </h1>

            {hero?.lead && (
              <p className="mt-6 max-w-4xl text-slate-400 text-base sm:text-lg leading-relaxed">
                {hero.lead}
              </p>
            )}

            {hero?.canonicalLatex && (
              <div className="mt-8 bg-[#0b1120] border border-slate-800/50 rounded-3xl p-6 sm:p-8 overflow-x-auto">
                <div className="text-blue-300 text-center min-w-max text-2xl sm:text-3xl">
                  <BlockMath math={hero.canonicalLatex} />
                </div>
              </div>
            )}
          </section>

          {/* SECTION: projection */}
          {projection && (
            <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
              <div className="lg:col-span-5 space-y-6">
                <div className="flex items-center gap-3 text-purple-400">
                  <Shrink size={22} />
                  <h2 className="text-2xl font-black uppercase tracking-tight">
                    {projection.heading}
                  </h2>
                </div>

                <div className="space-y-3 text-slate-400 leading-relaxed">
                  {projection.bullets?.map((b, i) => (
                    <div key={i} className="flex gap-3">
                      <div className="mt-2 w-1.5 h-1.5 bg-purple-500 rounded-full shrink-0" />
                      <div>
                        <span className="text-slate-200 font-bold">{b.k}</span>
                        <span className="text-slate-400"> — {b.v}</span>
                      </div>
                    </div>
                  ))}
                </div>

                {projection.latex && (
                  <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 shadow-xl overflow-x-auto">
                    <BlockMath math={projection.latex} />
                    <p className="mt-4 text-[11px] text-center text-slate-500 uppercase font-bold tracking-widest">
                      Shape & Projection Contract
                    </p>
                  </div>
                )}
              </div>

              <div className="lg:col-span-7 bg-[#0b1120] rounded-[2.5rem] border border-blue-500/20 shadow-[0_0_50px_rgba(59,130,246,0.1)] p-8 sm:p-10">
                <div className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-4">
                  Rules Preview
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                  {projection.rulesPreview?.map((r, i) => (
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

          {/* SECTION: equivalence */}
          {equivalence && (
            <section className="space-y-8">
              <div className="flex items-center gap-3 text-emerald-400">
                <ShieldCheck size={22} />
                <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                  {equivalence.heading}
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                {equivalence.cards?.map((c) => (
                  <div
                    key={c.id}
                    className="bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-xl group hover:border-emerald-500/30 transition"
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
                      <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 text-[11px] text-blue-200/70 font-mono italic overflow-x-auto">
                        <InlineMath math={c.metric} />
                      </div>
                    )}

                    {c.note && (
                      <div className="mt-4 text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                        {c.note}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </section>
          )}

          {/* SECTION: cost */}
          {cost && (
            <section className="bg-blue-600/5 border border-blue-500/20 rounded-[2.5rem] p-10 sm:p-12 shadow-xl text-center space-y-8">
              <div className="text-blue-500 font-mono text-xs font-black uppercase tracking-[0.4em]">
                {cost.heading}
              </div>

              {cost.latex && (
                <div className="text-2xl sm:text-4xl text-white font-black tracking-tight leading-relaxed">
                  <BlockMath math={cost.latex} />
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6 text-left">
                {cost.pills?.map((p, i) => (
                  <div
                    key={i}
                    className="bg-[#1e293b] border border-slate-800 rounded-2xl p-6"
                  >
                    <div className="text-[10px] font-black uppercase tracking-widest text-slate-500">
                      {p.tag}
                    </div>
                    <div className="mt-2 text-slate-100 font-black text-lg">
                      {p.title}
                    </div>
                    <p className="mt-2 text-slate-400 text-sm leading-relaxed">
                      {p.desc}
                    </p>
                  </div>
                ))}
              </div>

              {cost.foot && (
                <p className="text-slate-500 max-w-3xl mx-auto italic">
                  {cost.foot}
                </p>
              )}
            </section>
          )}
        </div>
      </main>
    </div>
  );
}