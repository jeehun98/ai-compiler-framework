// src/pages/HomePage.jsx
import React, { useState } from "react";
import { Link } from "react-router-dom";
import {
  Cpu, ShieldCheck, Layers, ArrowRight,
  Boxes, Sparkles, Waypoints, Menu,
} from "lucide-react";
import AppSidebar from "../components/AppSidebar.jsx";

export default function HomePage() {
  const [isSidebarOpen, setIsSidebarOpen] = useState(false);

  return (
    <div className="flex min-h-dvh bg-[#0f172a] text-slate-200 antialiased overflow-x-hidden">
      <AppSidebar
        isOpen={isSidebarOpen}
        onClose={() => setIsSidebarOpen(false)}
        activeOpId={null}
        quickOps={["AdamStep", "LayerNorm", "Softmax", "GEMM"]}
        version="v1.0.4 Stable"
      />

      <main className="flex-1 flex flex-col min-w-0">
        <header className="md:hidden fixed top-0 left-0 right-0 z-40 border-b border-slate-800 bg-[#0f172a]/90 backdrop-blur">
          <div className="flex items-center justify-between px-6 py-4">
            <div className="font-black text-blue-400 tracking-tighter uppercase">AICF Lab</div>
            <button onClick={() => setIsSidebarOpen(true)} className="p-2 rounded-xl border border-slate-700 bg-[#1e293b] text-slate-200">
              <Menu size={20} />
            </button>
          </div>
        </header>

        <div className="md:hidden h-[68px]" />

        <div className="flex-1 overflow-y-auto p-6 sm:p-10 space-y-16">
          {/* HERO SECTION: 수학적 본질로 메시지 변경 */}
          <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-12 shadow-2xl relative overflow-hidden">
            <div className="absolute -top-10 -right-10 text-[160px] font-black text-blue-500/5 pointer-events-none tracking-tighter uppercase">
              Semantic
            </div>

            <div className="flex items-center gap-2 text-blue-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
              <Cpu size={16} /> Pure Mathematics & Semantics
            </div>

            <h1 className="mt-6 text-5xl sm:text-6xl font-black tracking-tight leading-[1.1] text-white">
              데이터의 위상적 구조와
              <br />
              수학적 불변성을 보존하다
            </h1>

            <p className="mt-6 max-w-3xl text-slate-400 text-lg leading-relaxed">
              연산은 단순한 숫자의 나열이 아닙니다. AICF는 모든 연산을 
              <span className="text-slate-100 font-semibold italic"> "고차원 공간의 기하학적 투영"</span>으로 정의하며, 
              최적화 과정에서 정보의 위상(Topology)이 훼손되지 않도록 수학적 제약 조건을 설계합니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/theory"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-blue-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-blue-500 transition-all active:scale-95"
              >
                이론적 사양 보기 <ArrowRight size={18} />
              </Link>
              <Link
                to="/ops"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                연산 리포트 탐색
              </Link>
            </div>
          </section>

          {/* NARRATIVE: Pure Theory focus */}
          <section id="narrative" className="space-y-8">
            <div className="flex items-center gap-2 text-purple-400 font-black uppercase tracking-widest text-xs">
              <Waypoints size={16} /> Semantic Essence
            </div>

            <h2 className="text-4xl font-black tracking-tight text-white">
              수치적 근사에서,
              <br />
              논리적 동일성으로
            </h2>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {[
                {
                  k: "Definition",
                  t: "연산의 기하학적 정의",
                  sub: "Geometric Manifold",
                  icon: <Sparkles size={18} />,
                  p1: "모든 연산은 입력 매니폴드를 출력 매니폴드로 매핑하는 함수적 성질을 가집니다.",
                  p2: "우리는 이 매핑 과정에서 보존되어야 할 '불변량(Invariants)'을 식별합니다.",
                },
                {
                  k: "Invariance",
                  t: "불변성 기반 동일성",
                  sub: "Structural Identity",
                  icon: <Layers size={18} />,
                  p1: "수치가 다르더라도 데이터 간의 상대적 순위나 위상 구조가 같다면 이는 동일한 연산입니다.",
                  p2: "이 정의를 통해 근사 계산의 수학적 정당성을 확보합니다.",
                },
                {
                  k: "Constraint",
                  t: "의미론적 제약",
                  sub: "Logic Anchoring",
                  icon: <ShieldCheck size={18} />,
                  p1: "컴파일러는 수치적 오차가 아닌 '의미적 손실'을 비용 모델로 삼아 작동합니다.",
                  p2: "수학적 본질이 훼손되는 지점을 최적화의 한계선으로 설정합니다.",
                },
              ].map((s) => (
                <div key={s.k} className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl hover:border-blue-500/40 transition group">
                  <div className="flex items-center gap-2 text-slate-500 font-mono text-[10px] uppercase tracking-[0.25em] font-black">
                    {s.icon} {s.k}
                  </div>
                  <div className="mt-4">
                    <div className="text-blue-100 font-black text-xl tracking-tight leading-tight uppercase">{s.t}</div>
                    <div className="text-blue-500/60 font-mono text-[11px] font-bold uppercase tracking-wider mt-1">{s.sub}</div>
                  </div>
                  <p className="mt-6 text-slate-400 leading-relaxed text-[15px]">{s.p1}</p>
                  <p className="mt-3 text-slate-500 leading-relaxed text-[14px] italic">{s.p2}</p>
                </div>
              ))}
            </div>
          </section>

          {/* 신규 섹션: Execution Strategy (Plan Compiler) */}
          <section className="space-y-10 py-10">
            <div className="flex items-center gap-2 text-emerald-400 font-black uppercase tracking-widest text-xs">
              <Boxes size={16} /> Strategy Orchestration
            </div>

            <div className="flex flex-col lg:flex-row gap-12 items-start">
              <div className="lg:w-1/2 space-y-6">
                <h2 className="text-4xl font-black tracking-tight text-white leading-tight">
                  코드 생성을 넘어, <br />
                  <span className="text-emerald-400">최적의 실행 계획</span>을 컴파일하다
                </h2>
                <p className="text-slate-400 text-lg leading-relaxed">
                  AICF는 단순히 소스 코드를 출력하는 번역기가 아닙니다. <br/>
                  수학적 그래프를 분석하여 <strong>Kernel Registry</strong>에서 가장 적합한 연주자를 고르고, 메모리 레이아웃과 병렬 처리 정책을 결정하는 
                  <span className="text-slate-100 font-bold"> 'Execution Planner'</span>입니다.
                </p>
                
                <div className="grid grid-cols-2 gap-4 pt-4">
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="text-emerald-400 font-black text-2xl mb-1">0ms</div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-tighter">Zero JIT Overhead</div>
                  </div>
                  <div className="p-5 rounded-2xl bg-slate-800/50 border border-slate-700">
                    <div className="text-emerald-400 font-black text-2xl mb-1">Adaptive</div>
                    <div className="text-slate-400 text-xs uppercase font-bold tracking-tighter">Variant Selection</div>
                  </div>
                </div>
              </div>

              {/* 시각적 아키텍처 카드 */}
              <div className="lg:w-1/2 w-full bg-[#0b1120] border border-slate-800 rounded-[3rem] p-8 relative overflow-hidden group">
                <div className="absolute inset-0 bg-emerald-500/5 opacity-0 group-hover:opacity-100 transition-opacity" />
                <div className="space-y-6 relative">
                  <div className="flex items-center justify-between border-b border-slate-800 pb-4 text-xs font-mono text-slate-500 uppercase tracking-widest">
                    <span>Execution Plan Workflow</span>
                    <span className="text-emerald-500/50">Plan Compiler v1.0</span>
                  </div>
                  
                  <div className="space-y-4">
                    {[
                      { label: "Semantic Graph Analysis", detail: "Bitmask Pattern Matching", color: "bg-blue-500" },
                      { label: "Kernel Variant Selection", detail: "Registry Lookup (Pre-built)", color: "bg-indigo-500" },
                      { label: "Memory Layout Finalization", detail: "Launch Config & Stride Binding", color: "bg-emerald-500" }
                    ].map((step, i) => (
                      <div key={i} className="flex items-center gap-4">
                        <div className={`w-2 h-12 ${step.color} rounded-full`} />
                        <div>
                          <div className="text-white font-bold text-sm">{step.label}</div>
                          <div className="text-slate-500 text-xs font-mono mt-0.5">{step.detail}</div>
                        </div>
                      </div>
                    ))}
                  </div>

                  <Link to="/pipeline" className="mt-8 flex items-center justify-center w-full py-4 rounded-xl border border-emerald-500/30 text-emerald-400 font-bold text-xs uppercase tracking-widest hover:bg-emerald-500 hover:text-[#0b1120] transition-all">
                    실행 계획 생성 과정 상세 보기
                  </Link>
                </div>
              </div>
            </div>
          </section>
        </div>
      </main>
    </div>
  );
}