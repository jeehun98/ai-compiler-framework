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
  Info
} from "lucide-react";

import AppSidebar from "../components/AppSidebar.jsx";
import { theoryByOpId, theoryOpIds } from "../data/theory/index.js";

// Helper Component for equivalence cards
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
  const activeOpId = searchParams.get("op");
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  useEffect(() => {
    setIsSidebarOpen(false);
  }, [activeOpId]);

  const isMain = !activeOpId;
  const spec = activeOpId ? theoryByOpId[activeOpId] : null;

  // Render Error if op requested but not found
  if (activeOpId && !spec) {
    return (
      <div className="p-10 text-blue-400 bg-[#0f172a] min-h-screen flex flex-col items-center justify-center font-mono">
        <div className="animate-pulse mb-4 text-2xl font-black uppercase">Theory Not Found</div>
        <div className="text-slate-500 text-sm">data/theory에 "{activeOpId}" 스펙이 없습니다.</div>
        <Link to="/theory" className="mt-6 px-4 py-2 rounded-xl bg-blue-600 text-white font-bold">Back to Guide</Link>
      </div>
    );
  }

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
          <button onClick={() => setIsSidebarOpen(true)} className="p-2 rounded-xl border border-slate-700 bg-[#1e293b]">
            <Menu size={18} />
          </button>
        </div>
      </header>

      <AppSidebar isOpen={isSidebarOpen} onClose={() => setIsSidebarOpen(false)} activeOpId={activeOpId} />

      <main className="flex-1 flex flex-col min-w-0">
        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-14 pb-32 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          
          {isMain ? (
            /* ============================================================
               CASE 1: THEORY MAIN PAGE (INTERPRETATION GUIDE)
               ============================================================ */
            <div className="max-w-5xl mx-auto space-y-20 animate-in fade-in duration-700">
              {/* Hero Section */}
              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-16 shadow-2xl relative overflow-hidden">
                <div className="absolute -top-10 -right-10 text-[140px] font-black text-blue-500/5 pointer-events-none">THEORY</div>
                <div className="flex items-center gap-2 text-blue-500 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
                  <BookOpen size={16} /> Interpretation Guide
                </div>
                <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                  Mathematical Semantics <br/><span className="text-blue-500 text-3xl sm:text-5xl">of Operators</span>
                </h1>
                <p className="mt-8 text-slate-400 text-lg sm:text-xl leading-relaxed max-w-3xl">
                  Theory는 각 연산을 단순한 코드 조각이 아닌, <strong>특정한 구조를 보존하거나 변환하는 함수(Transformation)</strong>로 정의합니다. 
                  이 페이지는 각 Theory 문서를 어떻게 해석해야 하는지 설명하는 가이드입니다.
                </p>
              </section>

              {/* Scope & Intent */}
              <section className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-[#0b1120] border border-blue-500/20 rounded-[2rem] p-8">
                  <div className="flex items-center gap-3 text-blue-400 mb-6">
                    <Target size={24} />
                    <h2 className="text-xl font-black uppercase">What Theory Covers</h2>
                  </div>
                  <ul className="space-y-4 text-slate-300">
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-blue-500 rounded-full shrink-0" />
                      <span><strong>Mathematical Definition:</strong> 연산을 함수 f: X to Y 로 정의하고 성질을 명시합니다.</span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-blue-500 rounded-full shrink-0" />
                      <span><strong>Geometric Interpretation:</strong> 벡터 공간에서 수행하는 기하학적 변환을 설명합니다.</span>
                    </li>
                    <li className="flex gap-3 text-sm sm:text-base">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-blue-500 rounded-full shrink-0" />
                      <span><strong>Structural Invariants:</strong> 연산 후에도 보존되는 핵심 구조(거리, 순서 등)를 정의합니다.</span>
                    </li>
                  </ul>
                </div>

                <div className="bg-[#0b1120] border border-red-500/10 rounded-[2rem] p-8">
                  <div className="flex items-center gap-3 text-red-400 mb-6">
                    <XCircle size={24} />
                    <h2 className="text-xl font-black uppercase text-slate-300">What It Does NOT Contain</h2>
                  </div>
                  <ul className="space-y-4 text-slate-500 italic">
                    <li className="flex gap-3 line-through decoration-slate-700">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-red-900/50 rounded-full shrink-0" />
                      <span>구현 방식, 커널 구조 및 하드웨어 최적화 전략</span>
                    </li>
                    <li className="flex gap-3 line-through decoration-slate-700">
                      <div className="mt-1.5 w-1.5 h-1.5 bg-red-900/50 rounded-full shrink-0" />
                      <span>컴파일러 정책 및 실행 그래프 최적화</span>
                    </li>
                  </ul>
                  <p className="mt-10 p-4 bg-slate-900/50 rounded-xl border border-slate-800 text-[12px] text-slate-400 leading-relaxed font-medium">
                    <span className="text-blue-400 font-bold block mb-1">NOTE: Relationship with Ops Explorer</span>
                    Theory는 <strong>'무엇(What)'</strong>을 계산하는지 설명하고, Ops Explorer는 <strong>'어떻게(How)'</strong> 계산하는지 보여줍니다.
                  </p>
                </div>
              </section>

              {/* Component Structure */}
              <section className="space-y-8">
                <div className="flex items-center gap-3 text-emerald-400">
                  <Layers size={24} />
                  <h2 className="text-2xl font-black uppercase tracking-tight text-white">Theory Spec Structure</h2>
                </div>
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
                  {[
                    { id: "01", title: "Definition", desc: "연산의 수학적 공식과 입출력 도메인 정의", icon: Binary },
                    { id: "02", title: "Geometry", desc: "고차원 공간 내에서의 기하학적 변환 의미", icon: Shrink },
                    { id: "03", title: "Invariants", desc: "연산 이후에도 유지되는 수학적 본질", icon: ShieldCheck },
                    { id: "04", title: "Equivalence", desc: "동일한 의미를 가지는 수식적 조건", icon: Waypoints },
                  ].map((item) => (
                    <div key={item.id} className="bg-[#1e293b] p-8 rounded-[2rem] border border-slate-800 hover:border-blue-500/30 transition shadow-lg group">
                      <item.icon className="text-blue-500 mb-6 group-hover:scale-110 transition-transform" size={28} />
                      <div className="text-[10px] font-mono text-slate-600 mb-1 font-bold">SECTION {item.id}</div>
                      <h3 className="font-black text-white uppercase text-lg mb-2">{item.title}</h3>
                      <p className="text-slate-400 text-sm leading-relaxed">{item.desc}</p>
                    </div>
                  ))}
                </div>
              </section>

              {/* Core Operators Links */}
              <section className="bg-blue-600/5 border border-blue-500/20 rounded-[3rem] p-12 text-center">
                <h2 className="text-2xl font-black text-white uppercase mb-8">Operators Covered in Theory</h2>
                <div className="flex flex-wrap justify-center gap-4">
                  {theoryOpIds.map((id) => (
                    <Link
                      key={id}
                      to={`/theory?op=${id}`}
                      className="px-8 py-4 bg-[#0f172a] border border-slate-700 rounded-2xl text-blue-300 font-black hover:border-blue-500 hover:text-white transition shadow-xl uppercase tracking-wider"
                    >
                      {id}
                    </Link>
                  ))}
                </div>
                <p className="mt-10 text-slate-500 italic text-sm max-w-2xl mx-auto">
                  "Theory는 모든 연산을 다루지 않습니다. 수학적으로 핵심적인 기초 구조를 형성하는 연산만을 엄선하여 포함합니다."
                </p>
              </section>
            </div>
          ) : (
            /* ============================================================
               CASE 2: SPECIFIC THEORY DETAIL PAGE
               ============================================================ */
            <div className="space-y-14 animate-in slide-in-from-bottom-4 duration-500">
              {/* DETAIL HERO */}
              <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-12 shadow-2xl relative overflow-hidden">
                <div className="absolute -top-10 -right-10 text-[140px] sm:text-[180px] font-black text-blue-500/5 pointer-events-none tracking-tighter uppercase">
                  {spec.id}
                </div>
                <div className="flex items-center gap-2 text-blue-500 font-mono text-[10px] font-black uppercase tracking-[0.35em]">
                  <Waypoints size={14} /> {spec.subtitle || "Theory Spec"}
                </div>
                <h1 className="mt-4 text-4xl sm:text-6xl font-black tracking-tight text-white leading-[1.05]">
                  {spec.title}
                </h1>
                {spec.hero?.lead && (
                  <p className="mt-6 max-w-4xl text-slate-400 text-base sm:text-lg leading-relaxed italic">
                    {spec.hero.lead}
                  </p>
                )}
                {spec.hero?.canonicalLatex && (
                  <div className="mt-8 bg-[#0b1120] border border-slate-800/50 rounded-3xl p-6 sm:p-8 overflow-x-auto shadow-inner">
                    <div className="text-blue-300 text-center min-w-max text-2xl sm:text-3xl">
                      <BlockMath math={spec.hero.canonicalLatex} />
                    </div>
                  </div>
                )}
              </section>

              {/* SECTION: Projection (Geometry) */}
              {spec.sections?.projection && (
                <section className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
                  <div className="lg:col-span-5 space-y-6">
                    <div className="flex items-center gap-3 text-purple-400">
                      <Shrink size={22} />
                      <h2 className="text-2xl font-black uppercase tracking-tight">Geometric Interpretation</h2>
                    </div>
                    <div className="space-y-3 text-slate-400 leading-relaxed">
                      {spec.sections.projection.bullets?.map((b, i) => (
                        <div key={i} className="flex gap-3">
                          <div className="mt-2 w-1.5 h-1.5 bg-purple-500 rounded-full shrink-0" />
                          <div><span className="text-slate-200 font-bold">{b.k}</span> — {b.v}</div>
                        </div>
                      ))}
                    </div>
                    {spec.sections.projection.latex && (
                      <div className="bg-[#1e293b] p-6 rounded-3xl border border-slate-800 shadow-xl overflow-x-auto">
                        <BlockMath math={spec.sections.projection.latex} />
                        <p className="mt-4 text-[11px] text-center text-slate-500 uppercase font-bold tracking-widest">Projection Contract</p>
                      </div>
                    )}
                  </div>
                  <div className="lg:col-span-7 bg-[#0b1120] rounded-[2.5rem] border border-blue-500/20 p-8">
                    <div className="text-[10px] font-black uppercase tracking-widest text-slate-500 mb-4 flex items-center gap-2">
                      <Info size={12}/> Property Preview
                    </div>
                    <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                      {spec.sections.projection.rulesPreview?.map((r, i) => (
                        <div key={i} className="bg-[#0f172a] border border-slate-800 rounded-2xl p-5 hover:border-blue-500/30 transition">
                          <div className="text-slate-200 font-black uppercase tracking-tight">{r.k}</div>
                          <p className="mt-2 text-sm text-slate-400 leading-relaxed">{r.v}</p>
                        </div>
                      ))}
                    </div>
                  </div>
                </section>
              )}

              {/* SECTION: Equivalence & Invariance */}
              {spec.sections?.equivalence && (
                <section className="space-y-8">
                  <div className="flex items-center gap-3 text-emerald-400">
                    <ShieldCheck size={22} />
                    <h2 className="text-2xl font-black uppercase tracking-tight text-white">Invariants & Equivalence</h2>
                  </div>
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                    {spec.sections.equivalence.cards?.map((c) => (
                      <div key={c.id} className="bg-[#1e293b] p-8 rounded-[2.5rem] border border-slate-800 shadow-xl group hover:border-emerald-500/30 transition">
                        <div className="flex items-center justify-between mb-6">
                          <IconBadge icon={c.icon} />
                          <span className="text-[10px] font-mono text-slate-600 font-bold tracking-widest uppercase">Rule {c.id}</span>
                        </div>
                        <h3 className="text-xl font-black text-white mb-3 italic uppercase tracking-tighter">{c.title}</h3>
                        <p className="text-slate-400 text-sm leading-relaxed mb-5">{c.desc}</p>
                        {c.metric && (
                          <div className="bg-[#0f172a] border border-slate-800 rounded-2xl p-4 text-[11px] text-blue-200/70 font-mono italic overflow-x-auto">
                            <InlineMath math={c.metric} />
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                </section>
              )}

              {/* FOOTER NAV */}
              <div className="pt-10 border-t border-slate-800 flex justify-center">
                <Link to="/theory" className="flex items-center gap-2 text-blue-400 font-black uppercase text-sm hover:text-white transition">
                  <BookOpen size={16} /> Back to Theory Guide
                </Link>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}