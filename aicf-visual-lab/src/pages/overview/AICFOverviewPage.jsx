import React from "react";
import { Link } from "react-router-dom";
import {
  BrainCircuit,
  HardDrive,
  ArrowRight,
  Layers,
  Activity,
  Workflow,
} from "lucide-react";

export default function AICFOverviewPage() {
  return (
    <div className="min-h-dvh bg-[#070b14] text-slate-200 antialiased selection:bg-blue-500/30">
      <main className="mx-auto max-w-7xl px-6 py-12 sm:px-10 sm:py-20">
        {/* Hero Section */}
        <section className="relative overflow-hidden rounded-[3rem] border border-slate-800 bg-[#0f172a] p-10 sm:p-20 shadow-2xl">
          <div className="absolute -right-20 -top-20 h-96 w-96 rounded-full bg-blue-600/10 blur-[100px]" />
          <div className="absolute -left-20 -bottom-20 h-96 w-96 rounded-full bg-emerald-600/10 blur-[100px]" />

          <div className="relative z-10">
            <div className="flex items-center gap-3 text-blue-400 font-mono text-sm uppercase tracking-[0.4em] font-black">
              <span className="h-px w-8 bg-blue-500/50" />
              AI Compiler Framework
            </div>

            <h1 className="mt-8 text-5xl sm:text-8xl font-black tracking-tighter leading-[0.9] text-white">
              OPTIMIZATION <br />
              <span className="text-transparent bg-clip-text bg-gradient-to-r from-slate-200 to-slate-500">
                HAS TWO AXES
              </span>
            </h1>

            <p className="mt-10 max-w-3xl text-xl sm:text-2xl text-slate-400 leading-relaxed font-medium">
              AICF는 최적화를 하나의 기술이 아니라
              <br className="hidden sm:block" />
              <span className="text-cyan-400"> 같은 의미를 더 간결한 계산 구조로 실현하는 문제</span>와
              <span className="text-emerald-400"> 같은 결과를 더 적은 데이터 이동으로 실현하는 문제</span>
              로 나누어 다룹니다.
            </p>

            <div className="mt-10 flex flex-wrap gap-4">
              <Link
                to="/compute"
                className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-blue-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-blue-500 transition-all active:scale-95"
              >
                Compute Domain 보기 <ArrowRight size={18} />
              </Link>

              <Link
                to="/memory"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                Memory Domain 보기
              </Link>
            </div>
          </div>
        </section>

        {/* Two Pillars */}
        <section className="mt-12 grid grid-cols-1 lg:grid-cols-2 gap-8">
          {/* Compute Optimization */}
          <Link
            to="/compute"
            className="group relative overflow-hidden rounded-[2.5rem] border border-cyan-500/20 bg-[#111827] p-10 transition-all"
          >
            <div className="relative z-10">
              <div className="flex items-center justify-between">
                <div className="rounded-2xl bg-cyan-500/10 p-4 text-cyan-400 group-hover:scale-110 transition-transform">
                  <BrainCircuit size={40} strokeWidth={1.5} />
                </div>
                <ArrowRight
                  className="text-slate-600 group-hover:text-cyan-400 group-hover:translate-x-2 transition-all"
                  size={32}
                />
              </div>

              <h2 className="mt-12 text-4xl font-black text-white group-hover:text-cyan-500 transition-colors">
                Compute <br /> Optimization
              </h2>

              <p className="mt-6 text-lg text-slate-400 leading-relaxed">
                “같은 의미를 더 간결하게 계산할 수 있는가?”
                <br />
                의미를 보존하면서 계산 구조를 더 단순하고 효율적으로 바꿉니다.
                <br />
                semantic contract, equivalent reduction, approximation boundary를 다룹니다.
              </p>

              <div className="mt-10 flex flex-wrap gap-2">
                {[
                  "Semantic Contract",
                  "Equivalent Reduction",
                  "Approximation Boundary",
                ].map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 rounded-full bg-cyan-500/5 border border-cyan-500/20 text-[10px] font-black uppercase tracking-widest text-cyan-500/80"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            <div className="absolute right-0 bottom-0 opacity-5 group-hover:opacity-10 transition-opacity">
              <BrainCircuit size={240} />
            </div>
          </Link>

          {/* Memory Optimization */}
          <Link
            to="/memory"
            className="group relative overflow-hidden rounded-[2.5rem] border border-emerald-500/20 bg-[#111827] p-10 transition-all"
          >
            <div className="relative z-10">
              <div className="flex items-center justify-between">
                <div className="rounded-2xl bg-emerald-500/10 p-4 text-emerald-400 group-hover:scale-110 transition-transform">
                  <HardDrive size={40} strokeWidth={1.5} />
                </div>
                <ArrowRight
                  className="text-slate-600 group-hover:text-emerald-400 group-hover:translate-x-2 transition-all"
                  size={32}
                />
              </div>

              <h2 className="mt-12 text-4xl font-black text-white group-hover:text-emerald-500 transition-colors">
                Memory <br /> Optimization
              </h2>
              
              <p className="mt-6 text-lg text-slate-400 leading-relaxed">
                “같은 결과를 더 적은 이동으로 실현할 수 있는가?”
                <br />
                어떤 값은 local state로 유지하고, 어떤 intermediate는 제거하거나 다시 계산합니다.
                <br />
                residency, streaming reduction, rematerialization, traffic-aware planning을 다룹니다.
              </p>

              <div className="mt-10 flex flex-wrap gap-2">
                {[
                  "On-Chip Residency",
                  "Streaming Reduction",
                  "Traffic-Aware Planning",
                ].map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 rounded-full bg-emerald-500/5 border border-emerald-500/20 text-[10px] font-black uppercase tracking-widest text-emerald-500/80"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            <div className="absolute right-0 bottom-0 opacity-5 group-hover:opacity-10 transition-opacity">
              <HardDrive size={240} />
            </div>
          </Link>
        </section>

        {/* Integration Layer */}
        <section className="mt-20">
          <div className="rounded-[2.5rem] border border-slate-800 bg-[#0b1120] p-10 sm:p-14 relative overflow-hidden">
            <div className="absolute inset-0 bg-gradient-to-r from-blue-500/5 via-transparent to-emerald-500/5" />

            <div className="relative z-10 flex flex-col items-center text-center">
              <div className="rounded-full bg-blue-500 p-3 shadow-[0_0_20px_rgba(59,130,246,0.5)]">
                <Layers size={24} className="text-white" />
              </div>

              <h3 className="mt-8 text-3xl sm:text-4xl font-black text-white">
                AICF Execution Planner
              </h3>

              <p className="mt-5 max-w-3xl text-slate-400 text-lg leading-relaxed">
                Compute 축의 의미 제약과 Memory 축의 물리 제약을 함께 평가하여
                <br className="hidden sm:block" />
                최종 execution path와 kernel plan을 결정합니다.
              </p>

              <div className="mt-10 grid grid-cols-1 md:grid-cols-3 gap-4 w-full max-w-5xl">
                {[
                  {
                    icon: <BrainCircuit size={18} />,
                    title: "Semantic Analysis",
                    desc: "무엇이 반드시 보존되어야 하는가",
                    accent: "text-cyan-400 border-cyan-500/20 bg-cyan-500/5",
                  },
                  {
                    icon: <Workflow size={18} />,
                    title: "Path Selection",
                    desc: "어떤 실행 구조가 가장 타당한가",
                    accent: "text-blue-400 border-blue-500/20 bg-blue-500/5",
                  },
                  {
                    icon: <HardDrive size={18} />,
                    title: "Memory Planning",
                    desc: "어떤 state를 local하게 유지할 수 있는가",
                    accent: "text-emerald-400 border-emerald-500/20 bg-emerald-500/5",
                  },
                ].map((item) => (
                  <div
                    key={item.title}
                    className="rounded-[1.75rem] border border-slate-800 bg-[#111827] p-6 text-left"
                  >
                    <div
                      className={`inline-flex items-center gap-2 rounded-xl border px-3 py-2 text-xs font-black uppercase tracking-widest ${item.accent}`}
                    >
                      {item.icon}
                      {item.title}
                    </div>
                    <p className="mt-4 text-slate-400 leading-relaxed text-sm">
                      {item.desc}
                    </p>
                  </div>
                ))}
              </div>

              <div className="mt-10 flex flex-wrap justify-center gap-4">
                <Link
                  to="/memory/pipeline"
                  className="inline-flex items-center gap-2 px-7 py-4 rounded-2xl bg-emerald-600 text-white font-bold text-sm uppercase tracking-widest shadow-lg hover:bg-emerald-500 transition-all active:scale-95"
                >
                  Residency Pipeline 보기 <ArrowRight size={18} />
                </Link>

                <Link
                  to="/ops"
                  className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
                >
                  Ops Explorer 보기
                </Link>
              </div>
            </div>
          </div>
        </section>

        {/* Bottom navigation */}
        <section className="mt-14 text-center">
          <Link
            to="/analysis"
            className="group inline-flex items-center gap-3 text-slate-500 hover:text-white transition-colors"
          >
            <span className="font-mono text-xs uppercase tracking-[0.3em]">
              Go to kernel analysis
            </span>
            <Activity size={16} className="group-hover:animate-pulse" />
          </Link>
        </section>
      </main>
    </div>
  );
}