// src/pages/HomePage.jsx
import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Cpu,
  ShieldCheck,
  Layers,
  ArrowRight,
  Boxes,
  Sparkles,
  Waypoints,
  Menu,
  X,
  Terminal,
  LayoutDashboard
} from "lucide-react";

export default function HomePage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex h-screen bg-[#0f172a] text-slate-200 antialiased overflow-hidden">
      
      {/* GLOBAL SIDEBAR - OpsPage와 동일한 구성 */}
      <aside
        className={`
          fixed md:static inset-y-0 left-0 z-50 
          w-[85vw] max-w-[320px] md:w-80
          bg-[#1e293b] border-r border-slate-800
          flex flex-col shadow-2xl transition-transform duration-300
          ${isSidebarOpen ? 'translate-x-0' : '-translate-x-full md:translate-x-0'}
        `}
      >
        {/* Logo Area */}
        <div className="p-6 border-b border-slate-800 bg-[#0f172a]/50">
          <Link to="/" className="flex items-center gap-3 group">
            <div className="bg-blue-600 p-2 rounded-xl group-hover:bg-blue-500 transition">
              <Cpu size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-lg font-black tracking-tight text-white leading-none">AICF LAB</h1>
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">v1.0.4 Stable</span>
            </div>
          </Link>
        </div>

        {/* Global Menu Navigation */}
        <nav className="p-4 flex-1 space-y-1">
          <p className="px-3 text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">Navigation</p>
          <Link 
            to="/" 
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl bg-blue-600/10 text-blue-400 font-bold text-sm border border-blue-500/20"
          >
            <LayoutDashboard size={18} /> Dashboard
          </Link>
          <Link 
            to="/ops" 
            className="flex items-center gap-3 px-3 py-2.5 rounded-xl text-slate-400 hover:bg-slate-800 hover:text-white transition font-bold text-sm"
          >
            <Terminal size={18} /> Ops Explorer
          </Link>
          
          <div className="pt-8 px-3">
             <p className="text-[10px] font-black text-slate-500 uppercase tracking-widest mb-4">Quick Links</p>
             <div className="space-y-3">
                {["AdamStep", "LayerNorm", "Softmax"].map(op => (
                  <Link 
                    key={op}
                    to={`/ops?op=${op}`}
                    className="block text-xs font-semibold text-slate-500 hover:text-blue-400 transition"
                  >
                    # {op} Trace
                  </Link>
                ))}
             </div>
          </div>
        </nav>

        {/* Sidebar Footer */}
        <div className="p-6 border-t border-slate-800 text-[10px] text-slate-600 font-medium">
          © 2026 AICF Compiler Team. <br/> Semantic Preserving Engine.
        </div>
      </aside>

      {/* MAIN CONTENT AREA */}
      <main className="flex-1 flex flex-col min-w-0 overflow-hidden">
        
        {/* Mobile Header */}
        <header className="md:hidden flex items-center justify-between px-6 py-4 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="font-black text-blue-500 tracking-tighter uppercase">AICF Lab</div>
          <button onClick={() => setIsSidebarOpen(true)} className="p-2 bg-slate-800 rounded-lg text-slate-200">
            <Menu size={20} />
          </button>
        </header>

        {/* Scrollable Content */}
        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-16">
          
          {/* HERO SECTION */}
          <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-12 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[160px] font-black text-blue-500/5 pointer-events-none tracking-tighter">
              AICF
            </div>

            <div className="flex items-center gap-2 text-blue-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <Cpu size={16} /> AI Compiler Framework
            </div>

            <h1 className="mt-6 text-5xl sm:text-6xl font-black tracking-tight leading-[1.1] text-white">
              연산의 의미를 보존하는 방식으로
              <br />
              AI 실행을 설계하다
            </h1>

            <p className="mt-6 max-w-3xl text-slate-400 text-lg leading-relaxed">
              AICF는 연산을 <span className="text-slate-100 font-semibold">“구현 이름”</span>이 아니라 <span className="italic text-blue-300">의미를 가진 객체</span>로 정의하고,
              그 의미가 최적화 레이어를 지나도 유지되도록 구조를 설계합니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/ops"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-blue-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-blue-500 transition-all active:scale-95"
              >
                연산 정의 보기 <ArrowRight size={18} />
              </Link>

              <div className="flex gap-2">
                <a
                  href="#narrative"
                  className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
                >
                  흐름 보기
                </a>
              </div>
            </div>

            <div className="mt-12 flex items-center gap-2 text-emerald-400 font-bold bg-emerald-400/5 px-4 py-2 rounded-xl border border-emerald-400/10 text-xs uppercase tracking-widest w-fit">
              <ShieldCheck size={16} /> Semantic Preserving Architecture
            </div>
          </section>

          {/* NARRATIVE SECTION */}
          <section id="narrative" className="space-y-8">
            <div className="flex items-center gap-2 text-purple-400 font-black uppercase tracking-widest text-xs">
              <Waypoints size={16} /> Narrative
            </div>

            <h2 className="text-4xl font-black tracking-tight text-white">
              수식의 의미에서,
              <br />
              의미 보존형 실행 설계로
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {[
                {
                  k: "Step 1",
                  t: "수식은 정적이다",
                  sub: "The Ideal",
                  icon: <Sparkles size={18} />,
                  p1: "AI 모델은 y = f(x)라는 수학적 정의에서 출발한다.",
                  p2: "데이터 흐름은 하나의 논리로서 계산 그래프를 이룬다.",
                },
                {
                  k: "Step 2",
                  t: "최적화의 역설",
                  sub: "The Reality",
                  icon: <Layers size={18} />,
                  p1: "성능 최적화 과정에서 수학적 구조는 하드웨어 친화적 형태로 치환된다.",
                  p2: "이 과정에서 연산의 경계와 의미 단위가 흐려지기 쉽다.",
                },
                {
                  k: "Step 3",
                  t: "연산의 재정의",
                  sub: "The Approach",
                  icon: <ShieldCheck size={18} />,
                  p1: "의미가 구조적으로 정의되지 않았다는 점이 문제의 핵심이다.",
                  p2: "연산을 의미론적 객체로 고정하여 최적화를 설계한다.",
                },
              ].map((s) => (
                <div
                  key={s.k}
                  className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl hover:border-blue-500/40 transition group"
                >
                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] uppercase tracking-[0.25em] font-black">
                    {s.icon} {s.k}
                  </div>
                  <div className="mt-4">
                    <div className="text-blue-100 font-black text-xl tracking-tight leading-tight uppercase">
                      {s.t}
                    </div>
                    <div className="text-blue-500/60 font-mono text-[11px] font-bold uppercase tracking-wider mt-1">
                      {s.sub}
                    </div>
                  </div>
                  <p className="mt-6 text-slate-400 leading-relaxed text-[15px]">
                    {s.p1}
                  </p>
                  <p className="mt-3 text-slate-500 leading-relaxed text-[14px] italic">
                    {s.p2}
                  </p>
                </div>
              ))}
            </div>
          </section>

          {/* SYSTEM MAP SECTION */}
          <section id="architecture" className="space-y-8 pb-20">
            <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
              <Boxes size={16} /> System Map
            </div>
            <h2 className="text-4xl font-black tracking-tight text-white uppercase">
              Meaning → IR → Kernel
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {[
                { title: "Meaning (Contract)", icon: <ShieldCheck size={18} />, desc: "연산 정의와 허용 변형 규칙" },
                { title: "IR (Transform)", icon: <Layers size={18} />, desc: "의미 보존 조건 하에서 Rewrite 수행" },
                { title: "Kernel (Realization)", icon: <Cpu size={18} />, desc: "의미 계약을 실현하는 최적 커널 구성" },
              ].map((c) => (
                <div
                  key={c.title}
                  className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl"
                >
                  <div className="flex items-center gap-2 text-slate-400 font-mono text-[10px] uppercase tracking-[0.2em] font-black">
                    {c.icon} {c.title}
                  </div>
                  <p className="mt-6 text-slate-200 font-bold text-xl tracking-tight leading-snug">
                    {c.desc}
                  </p>
                </div>
              ))}
            </div>
          </section>
        </div>
      </main>

      {/* Mobile Overlay */}
      {isSidebarOpen && (
        <div 
          className="fixed inset-0 z-40 bg-black/60 md:hidden backdrop-blur-sm"
          onClick={() => setIsSidebarOpen(false)}
        />
      )}
    </div>
  );
}