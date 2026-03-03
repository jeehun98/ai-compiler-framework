import React, { useMemo, useState, useEffect } from "react";
import { Link, useSearchParams } from "react-router-dom";
import { InlineMath, BlockMath } from "react-katex";
import {
  Dna, Target, Binary, Waypoints,
  ArrowRightLeft, Shrink, Zap, ShieldCheck,
  Layers, ArrowUpRight, Menu
} from "lucide-react";

import AppSidebar from "../components/AppSidebar.jsx";
import { theoryByOpId, DEFAULT_THEORY_OP } from "../data/theory/index.js";

function iconByName(name) {
  if (name === "target") return <Target className="text-emerald-400" />;
  if (name === "binary") return <Binary className="text-blue-400" />;
  if (name === "arrow") return <ArrowRightLeft className="text-purple-400" />;
  return <ShieldCheck className="text-slate-400" />;
}

export default function TheoryPage() {
  const [searchParams, setSearchParams] = useSearchParams();
  const opFromUrl = searchParams.get("op") || DEFAULT_THEORY_OP;

  const [activeOp, setActiveOp] = useState(opFromUrl);
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);
  const [semanticMode, setSemanticMode] = useState("projection");

  useEffect(() => {
    if (opFromUrl !== activeOp) setActiveOp(opFromUrl);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [opFromUrl]);

  const spec = theoryByOpId[activeOp] || theoryByOpId[DEFAULT_THEORY_OP];

  const tabs = useMemo(() => ([
    { id: "projection", label: "Projection" },
    { id: "equivalence", label: "Equivalence" },
    { id: "cost", label: "Cost" },
  ]), []);

  const scrollTo = (id) => {
    setSemanticMode(id);
    document.getElementById(`sec-${id}`)?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const handleOpChange = (nextOp) => {
    setActiveOp(nextOp);
    setSearchParams({ op: nextOp }, { replace: true });
  };

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      {/* Sidebar unified */}
      <AppSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeOpId={activeOp}
        quickOps={["AdamStep", "LayerNorm", "Softmax", "GEMM"]}
      />

      {/* MAIN */}
      <main className="flex-1 flex flex-col min-w-0">
        {/* Mobile Header */}
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-5 py-4">
            <div className="font-black text-blue-400 tracking-tight uppercase">AICF Lab</div>
            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200"
              aria-label="Open sidebar"
            >
              <Menu size={18} />
            </button>
          </div>
        </header>
        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-16">
          {/* HEADER */}
          <section className="border-b border-slate-800 pb-10">
            <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black uppercase tracking-[0.4em] mb-4">
              <Dna size={14} /> {spec.subtitle || "AICF Semantic Philosophy"}
            </div>

            <div className="flex flex-col lg:flex-row lg:items-end justify-between gap-6">
              <div className="min-w-0">
                <h1 className="text-4xl sm:text-5xl font-black tracking-tighter text-white leading-tight break-words">
                  {spec.title}
                </h1>
                <p className="mt-6 text-slate-400 text-base sm:text-lg max-w-4xl leading-relaxed">
                  {spec.hero?.lead}
                </p>

                <div className="mt-8 flex flex-wrap items-center gap-3">
                  <Link
                    to={`/ops?op=${spec.id}`}
                    className="inline-flex items-center gap-2 px-5 py-3 rounded-2xl bg-emerald-500/10 border border-emerald-500/20 text-emerald-300 font-black text-xs uppercase tracking-widest hover:bg-emerald-500/20 transition"
                  >
                    <Zap size={16} /> Open in Ops Explorer <ArrowUpRight size={16} className="opacity-70" />
                  </Link>

                  <div className="flex flex-wrap gap-2">
                    {tabs.map((t) => (
                      <button
                        key={t.id}
                        onClick={() => scrollTo(t.id)}
                        className={[
                          "px-4 py-2 rounded-xl border text-xs font-black uppercase tracking-widest transition",
                          semanticMode === t.id
                            ? "bg-blue-600/15 border-blue-500/30 text-blue-300"
                            : "bg-[#1e293b] border-slate-800 text-slate-400 hover:text-white hover:border-slate-700",
                        ].join(" ")}
                      >
                        {t.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* op switcher (theory 전용) */}
              <div className="bg-[#1e293b] border border-slate-800 rounded-2xl p-4 shadow-xl w-full lg:w-[360px]">
                <div className="text-[10px] text-slate-500 font-black uppercase tracking-widest mb-2">
                  Theory Target Op
                </div>
                <div className="flex gap-2 flex-wrap">
                  {Object.keys(theoryByOpId).map((op) => {
                    const active = op === spec.id;
                    return (
                      <button
                        key={op}
                        onClick={() => handleOpChange(op)}
                        className={[
                          "px-3 py-2 rounded-xl border text-xs font-black uppercase tracking-widest transition",
                          active
                            ? "bg-blue-600/15 border-blue-500/30 text-blue-300"
                            : "bg-[#0f172a] border-slate-800 text-slate-400 hover:text-white hover:border-slate-700",
                        ].join(" ")}
                      >
                        {op}
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>
          </section>

          {/* CONTRACT */}
          <section className="grid grid-cols-1 lg:grid-cols-12 gap-6">
            <div className="lg:col-span-7 bg-[#1e293b] p-7 sm:p-8 rounded-[2.5rem] border border-slate-800 shadow-xl">
              <div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] font-black uppercase tracking-[0.35em] mb-4">
                <Layers size={14} /> Semantic Contract ({spec.id})
              </div>

              <div className="bg-[#0f172a] p-4 rounded-2xl border border-slate-800 overflow-x-auto scrollbar-hide">
                <div className="min-w-max text-blue-300 text-center">
                  <BlockMath math={spec.hero?.canonicalLatex || ""} />
                </div>
                <div className="mt-3 text-[10px] text-slate-500 font-mono uppercase tracking-widest text-center">
                  Canonical Form (Semantic Anchor)
                </div>
              </div>
            </div>

            <div className="lg:col-span-5 bg-[#0b1120] p-7 sm:p-8 rounded-[2.5rem] border border-blue-500/20 shadow-[0_0_50px_rgba(59,130,246,0.08)] relative overflow-hidden">
              <div className="flex items-center gap-2 text-blue-400 font-mono text-[10px] font-black uppercase tracking-[0.4em] mb-4">
                <Waypoints size={14} /> Visualization Placeholder
              </div>
              <div className="text-center space-y-4 z-10 relative">
                <div className="animate-pulse flex justify-center">
                  <Waypoints size={56} className="text-blue-500/50" />
                </div>
                <div className="font-mono text-[10px] text-blue-400 tracking-[0.5em] uppercase">
                  Visualizing Semantic Mapping
                </div>
              </div>
              <div className="absolute inset-0 opacity-70 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)]" />
            </div>
          </section>

          {/* PROJECTION */}
          <section id="sec-projection" className="grid grid-cols-1 lg:grid-cols-12 gap-10 items-start scroll-mt-24">
            <div className="lg:col-span-5 space-y-6">
              <div className="flex items-center gap-3 text-purple-400">
                <Shrink size={24} />
                <h3 className="text-2xl font-black uppercase tracking-tight">
                  {spec.sections?.projection?.heading || "Projection"}
                </h3>
              </div>

              <div className="space-y-3 text-slate-400 leading-relaxed">
                {(spec.sections?.projection?.bullets || []).map((b, i) => (
                  <div key={i} className="flex gap-3">
                    <div className="mt-1.5 w-1.5 h-1.5 bg-purple-500 rounded-full shrink-0" />
                    <span><strong>{b.k}:</strong> {b.v}</span>
                  </div>
                ))}
              </div>

              <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 shadow-xl">
                <div className="overflow-x-auto scrollbar-hide">
                  <div className="min-w-max">
                    <BlockMath math={spec.sections?.projection?.latex || ""} />
                  </div>
                </div>
                <p className="mt-4 text-[11px] text-center text-slate-500 uppercase font-bold tracking-widest">
                  K-Traversals as Semantic Probe
                </p>
              </div>

              {!!(spec.sections?.projection?.rulesPreview?.length) && (
                <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 shadow-xl">
                  <p className="text-[10px] text-slate-500 font-black uppercase tracking-widest mb-3">
                    Allowed Rewrite (Preview)
                  </p>
                  <div className="space-y-2 text-sm text-slate-300">
                    {spec.sections.projection.rulesPreview.map((r, i) => (
                      <div key={i} className="flex items-start justify-between gap-3 bg-[#0f172a] px-4 py-3 rounded-2xl border border-slate-800">
                        <span className="text-[10px] font-black uppercase tracking-widest text-slate-500 shrink-0">
                          {r.k}
                        </span>
                        <span className="text-sm text-slate-300 font-bold break-words">
                          {r.v}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>

            <div className="lg:col-span-7 bg-[#0b1120] aspect-video rounded-[3rem] border border-blue-500/20 shadow-[0_0_50px_rgba(59,130,246,0.08)] flex items-center justify-center relative overflow-hidden">
              <div className="text-center space-y-4 z-10">
                <div className="animate-pulse flex justify-center">
                  <Waypoints size={64} className="text-blue-500/50" />
                </div>
                <div className="font-mono text-[10px] text-blue-400 tracking-[0.5em] uppercase">
                  Projection View
                </div>
              </div>
              <div className="absolute inset-0 bg-[linear-gradient(to_right,#1e293b_1px,transparent_1px),linear-gradient(to_bottom,#1e293b_1px,transparent_1px)] bg-[size:40px_40px] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_50%,#000_70%,transparent_100%)]" />
            </div>
          </section>

          {/* EQUIVALENCE */}
          <section id="sec-equivalence" className="space-y-10 scroll-mt-24">
            <div className="flex items-center gap-3 text-emerald-400">
              <ShieldCheck size={24} />
              <h3 className="text-2xl font-black uppercase tracking-tight text-white">
                {spec.sections?.equivalence?.heading || "Equivalence"}
              </h3>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
              {(spec.sections?.equivalence?.cards || []).map((item) => (
                <div
                  key={item.id}
                  className="bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-xl group hover:border-emerald-500/30 transition"
                >
                  <div className="flex items-center justify-between mb-6">
                    <div className="p-3 bg-[#0f172a] rounded-2xl border border-slate-800">
                      {iconByName(item.icon)}
                    </div>
                    <span className="text-[10px] font-mono text-slate-600 font-bold tracking-widest uppercase">
                      Rule {item.id}
                    </span>
                  </div>

                  <h4 className="text-xl font-black text-white mb-4 italic uppercase tracking-tighter">
                    {item.title}
                  </h4>

                  <p className="text-slate-400 text-sm leading-relaxed mb-5">
                    {item.desc}
                  </p>

                  <div className="bg-[#0f172a] p-4 rounded-2xl border border-slate-800">
                    <p className="text-[9px] text-slate-500 font-black uppercase tracking-widest mb-2">
                      Verification Metric
                    </p>
                    <div className="text-[12px] text-blue-200/80 font-mono italic overflow-x-auto scrollbar-hide">
                      <div className="min-w-max">
                        <InlineMath math={item.metric} />
                      </div>
                    </div>
                    {item.note && <p className="mt-3 text-[10px] text-slate-500">{item.note}</p>}
                  </div>
                </div>
              ))}
            </div>
          </section>

          {/* COST */}
          <section
            id="sec-cost"
            className="bg-blue-600/5 border border-blue-500/20 rounded-[2.5rem] p-8 sm:p-12 shadow-xl text-center scroll-mt-24"
          >
            <p className="text-blue-500 font-mono text-xs font-black uppercase tracking-[0.4em] mb-6">
              {spec.sections?.cost?.heading || "Semantic Cost"}
            </p>

            <div className="text-xl sm:text-3xl text-white font-black tracking-tight leading-relaxed overflow-x-auto scrollbar-hide">
              <div className="min-w-max">
                <BlockMath math={spec.sections?.cost?.latex || ""} />
              </div>
            </div>

            <div className="mt-8 grid grid-cols-1 md:grid-cols-3 gap-4 text-left">
              {(spec.sections?.cost?.pills || []).map((p, i) => (
                <div key={i} className="bg-[#0f172a] p-5 rounded-2xl border border-slate-800">
                  <div className="flex items-center justify-between gap-3">
                    <p className="text-sm font-black text-white">{p.title}</p>
                    <span className="text-[10px] font-mono text-slate-500 bg-slate-900/60 border border-slate-800 px-2 py-1 rounded-lg">
                      {p.tag}
                    </span>
                  </div>
                  <p className="mt-3 text-sm text-slate-400 leading-relaxed">{p.desc}</p>
                </div>
              ))}
            </div>

            {spec.sections?.cost?.foot && (
              <p className="mt-10 text-slate-500 max-w-2xl mx-auto italic">
                {spec.sections.cost.foot}
              </p>
            )}
          </section>

          <style jsx="true">{`
            .scrollbar-hide::-webkit-scrollbar { display: none; }
            .scrollbar-hide { -ms-overflow-style: none; scrollbar-width: none; }
          `}</style>
        </div>
      </main>
    </div>
  );
}