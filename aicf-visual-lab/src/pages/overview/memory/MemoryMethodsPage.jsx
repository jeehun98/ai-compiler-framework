import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Database,
  ArrowUpRight,
  HardDrive,
  Menu,
} from "lucide-react";


import { memoryMethodCatalog } from "../../../data/memory/methodCatalog";
import MemorySidebar from "../../../components/layout/MemorySidebar.jsx";

export default function MemoryMethodsPage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      {/* GLOBAL SIDEBAR */}
      <MemorySidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        version="v1.0.6 Lab-Ready"
      />

      {/* MAIN CONTENT */}
      <main className="flex-1 flex flex-col min-w-0 font-sans">
        {/* Mobile Header */}
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-5 py-4">
            <Link to="/" className="flex items-center gap-2">
              <div className="bg-emerald-600 p-2 rounded-xl">
                <HardDrive size={18} className="text-white" />
              </div>
              <div className="leading-none">
                <div className="font-black text-emerald-400 tracking-tight">
                  AICF MEMORY
                </div>
                <div className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                  v1.0.6 Lab-Ready
                </div>
              </div>
            </Link>

            <button
              onClick={() => setIsSidebarOpen(true)}
              className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200 active:scale-95 transition"
              aria-label="Open sidebar"
              type="button"
            >
              <Menu size={18} />
            </button>
          </div>
        </header>

        {/* Mobile top spacer */}
        <div className="md:hidden h-[68px]" />

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-14 pb-32 bg-[linear-gradient(180deg,rgba(15,23,42,1),rgba(30,41,59,0.2))]">
          <div className="max-w-6xl mx-auto space-y-20 animate-in fade-in duration-700">
            {/* Hero Section */}
            <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 sm:p-16 shadow-2xl relative overflow-hidden">
              <div className="absolute -top-10 -right-10 text-[140px] font-black text-emerald-500/5 pointer-events-none">
                MCIR
              </div>

              <div className="flex items-center gap-2 text-emerald-400 font-mono text-xs font-black uppercase tracking-[0.3em] mb-6">
                <Database size={16} /> Architecture Pillars
              </div>

              <h1 className="text-4xl sm:text-6xl font-black tracking-tight text-white leading-tight">
                Optimization <br />
                <span className="text-transparent bg-clip-text bg-gradient-to-r from-emerald-400 to-cyan-400">
                  Methodologies
                </span>
              </h1>

              <p className="mt-8 text-slate-400 text-lg sm:text-xl leading-relaxed max-w-3xl font-light">
                AICF는 단순한 코드 최적화를 넘어, 하드웨어의 물리적 제약을 수학적 성질로 극복합니다.
                <br />
                아래의 4가지 핵심 기법은 메모리 벽(Memory Wall)을 허물고 연산 효율을 물리적 한계치까지
                끌어올리는 AICF의 기술적 기둥입니다.
              </p>
            </section>

            {/* Method Grid */}
            <section className="space-y-8">
              <div className="flex items-center gap-3 text-emerald-400">
                <Database size={24} />
                <h2 className="text-2xl font-black uppercase tracking-tight text-white">
                  Core Optimization Methods
                </h2>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                {memoryMethodCatalog.map((method) => {
                  const Icon = method.icon;

                  return (
                    <Link
                      key={method.id}
                      to={`/memory/methods/${method.id}`}
                      className={`group relative p-8 rounded-[2rem] bg-[#1e293b]/40 border ${method.color} transition-all duration-500 hover:-translate-y-2`}
                    >
                      <div className="absolute top-0 right-0 p-8 opacity-10 group-hover:opacity-20 transition-opacity">
                        <Icon className={method.iconColor} size={32} />
                      </div>

                      <div className="relative z-10">
                        <div className="mb-6 p-4 w-fit rounded-2xl bg-slate-900/50 border border-slate-700 group-hover:border-slate-500 transition-colors">
                          <Icon className={method.iconColor} size={32} />
                        </div>

                        <div className="space-y-2 mb-6">
                          <span className="text-xs font-black text-emerald-500/70 uppercase tracking-widest">
                            {method.category}
                          </span>
                          <h3 className="text-2xl font-black text-white group-hover:text-emerald-400 transition-colors">
                            {method.label}
                          </h3>
                        </div>

                        <p className="text-slate-400 text-sm leading-relaxed mb-8 group-hover:text-slate-300 transition-colors">
                          {method.desc}
                        </p>

                        <div className="flex flex-wrap gap-2 mb-8">
                          {method.tags.map((tag) => (
                            <span
                              key={tag}
                              className="px-3 py-1 rounded-full bg-slate-900 text-[10px] font-bold text-slate-500 border border-slate-800"
                            >
                              #{tag}
                            </span>
                          ))}
                        </div>

                        <div className="flex items-center gap-2 text-emerald-400 font-black text-xs uppercase tracking-widest opacity-0 group-hover:opacity-100 transition-all translate-x-[-10px] group-hover:translate-x-0">
                          Explore Tech Spec <ArrowUpRight size={14} />
                        </div>
                      </div>
                    </Link>
                  );
                })}
              </div>
            </section>

            {/* Philosophy Section */}
            <section className="bg-emerald-600/5 border border-emerald-500/20 rounded-[3rem] p-12 text-center">
              <h2 className="text-2xl font-black text-white uppercase mb-6">
                Mathematical Properties Become Physical Speed
              </h2>

              <p className="text-slate-500 text-sm max-w-2xl mx-auto leading-relaxed italic">
                "모든 최적화는 값의 성질을 정의하는 것에서 시작합니다.
                <br />
                AICF는 연산의 결합 법칙과 단조성을 이용하여 데이터가 칩 밖을 나가지 않아도 되는 수학적 증명을 코드에 주입합니다."
              </p>
            </section>
          </div>
        </div>
      </main>
    </div>
  );
}