import React from "react";
import { Link } from "react-router-dom";
import {
  BrainCircuit,
  HardDrive,
  ArrowRight,
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
                HAS THREE AXES
              </span>
            </h1>

            <p className="mt-10 max-w-4xl text-xl sm:text-2xl text-slate-400 leading-relaxed font-medium">
              AICF는 최적화를 하나의 기술이 아니라
              <br className="hidden sm:block" />
              <span className="text-cyan-400">
                {" "}
                의미를 보존하며 실행 경로를 설계하는 Compute
              </span>
              ,
              <br className="hidden sm:block" />
              <span className="text-emerald-400">
                {" "}
                데이터 이동의 물리 구조를 다루는 Memory
              </span>
              ,
              <br className="hidden sm:block" />
              <span className="text-violet-400">
                {" "}
                실제 실행과 관측을 연결하는 Lab
              </span>
              으로 나누어 다룹니다.
            </p>
          </div>
        </section>

        {/* Three Domains */}
        <section className="mt-12 grid grid-cols-1 xl:grid-cols-3 gap-8">
          {/* Compute */}
          <Link
            to="/compute"
            className="group relative overflow-hidden rounded-[2.5rem] border border-cyan-500/20 bg-[#111827] p-10 transition-all hover:border-cyan-500/40"
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
                “불변적 성질 안에서 최적의 실행 경로를 결정할 수 있는가?”
                <br />
                연산의 의미를 훼손하지 않으면서, 런타임 상황에 따라 실제 연산
                파라미터와 실행 경로를 동적으로 조정합니다.
              </p>

              <div className="mt-10 flex flex-wrap gap-2">
                {[
                  "Invariant Preservation",
                  "Runtime Adaptive",
                  "Path Selection",
                  "Parameter Tuning",
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

          {/* Memory */}
          <Link
            to="/memory"
            className="group relative overflow-hidden rounded-[2.5rem] border border-emerald-500/20 bg-[#111827] p-10 transition-all hover:border-emerald-500/40"
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
                “동일한 결과를 더 적은 데이터 이동으로 실현할 수 있는가?”
                <br />
                데이터의 물리적 배치와 재사용 구조를 최적화하여 하드웨어 제약
                내에서 데이터 이동 비용을 최소화합니다.
              </p>

              <div className="mt-10 flex flex-wrap gap-2">
                {[
                  "On-Chip Residency",
                  "Traffic-Aware Planning",
                  "Memory-Centric IR",
                  "Rematerialization",
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

          {/* Lab */}
          <Link
            to="/lab"
            className="group relative overflow-hidden rounded-[2.5rem] border border-violet-500/20 bg-[#111827] p-10 transition-all hover:border-violet-500/40"
          >
            <div className="relative z-10">
              <div className="flex items-center justify-between">
                <div className="rounded-2xl bg-violet-500/10 p-4 text-violet-400 group-hover:scale-110 transition-transform">
                  <Activity size={40} strokeWidth={1.5} />
                </div>
                <ArrowRight
                  className="text-slate-600 group-hover:text-violet-400 group-hover:translate-x-2 transition-all"
                  size={32}
                />
              </div>

              <h2 className="mt-12 text-4xl font-black text-white group-hover:text-violet-500 transition-colors">
                AICF <br /> Lab
              </h2>

              <p className="mt-6 text-lg text-slate-400 leading-relaxed">
                “설계된 최적화가 실제 실행에서 어떻게 드러나는가?”
                <br />
                runtime path, kernel behavior, memory residency를 함께 관측하고
                검증하며, AICF의 설계를 실험 가능한 구조로 연결합니다.
              </p>

              <div className="mt-10 flex flex-wrap gap-2">
                {[
                  "Execution Pipeline",
                  "Kernel Analysis",
                  "Runtime Tracing",
                  "Residency Validation",
                ].map((tag) => (
                  <span
                    key={tag}
                    className="px-3 py-1 rounded-full bg-violet-500/5 border border-violet-500/20 text-[10px] font-black uppercase tracking-widest text-violet-500/80"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>

            <div className="absolute right-0 bottom-0 opacity-5 group-hover:opacity-10 transition-opacity">
              <Workflow size={240} />
            </div>
          </Link>
        </section>
      </main>
    </div>
  );
}