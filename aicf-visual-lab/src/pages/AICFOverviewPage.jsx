// src/pages/AICFOverviewPage.jsx
import React from "react";
import { Link } from "react-router-dom";
import {
  BrainCircuit,
  HardDrive,
  ArrowRight,
  Sigma,
  Workflow,
} from "lucide-react";

export default function AICFOverviewPage() {
  return (
    <div className="min-h-dvh bg-[#0b1120] text-slate-200 antialiased">
      <main className="mx-auto max-w-7xl px-6 py-10 sm:px-10 sm:py-16">
        {/* Top */}
        <section className="border border-slate-800 bg-[#111827] rounded-[2.5rem] px-8 py-10 sm:px-12 sm:py-14 shadow-2xl overflow-hidden relative">
          <div className="absolute -top-8 -right-4 sm:-top-10 sm:right-2 text-[80px] sm:text-[140px] font-black tracking-tighter text-white/5 uppercase pointer-events-none">
            AICF
          </div>

          <div className="relative">
            <div className="text-blue-400 font-mono text-xs sm:text-sm uppercase tracking-[0.35em] font-black">
              AI Compiler Framework
            </div>

            <h1 className="mt-5 text-4xl sm:text-6xl font-black tracking-tight leading-[1.05] text-white">
              AI 최적화는
              <br />
              하나가 아니다
            </h1>

            <p className="mt-6 max-w-4xl text-slate-400 text-base sm:text-lg leading-relaxed">
              AICF는 최적화를 단일 기법으로 보지 않습니다. 하나는 모델 의미를
              유지하며 계산량을 다루는 <span className="text-cyan-400 font-semibold">Compute Optimization</span>,
              다른 하나는 하드웨어 위 실제 데이터 이동 병목을 제거하는{" "}
              <span className="text-emerald-400 font-semibold">Memory Optimization</span>
              입니다.
            </p>

            <div className="mt-8 flex flex-wrap gap-4">
              <Link
                to="/"
                className="inline-flex items-center gap-2 rounded-2xl bg-blue-600 px-6 py-4 text-sm font-black uppercase tracking-widest text-white hover:bg-blue-500 transition"
              >
                기존 홈으로 이동 <ArrowRight size={16} />
              </Link>

              <Link
                to="/theory"
                className="inline-flex items-center gap-2 rounded-2xl border border-slate-700 px-6 py-4 text-xs font-black uppercase tracking-widest text-slate-300 hover:bg-slate-800 transition"
              >
                Theory 보기
              </Link>
            </div>
          </div>
        </section>

        {/* Definition */}
        <section className="mt-10">
          <div className="rounded-[2rem] border border-slate-800 bg-[#111827] px-8 py-8 sm:px-10">
            <div className="text-slate-500 font-mono text-[11px] uppercase tracking-[0.3em] font-black">
              Definition
            </div>
            <p className="mt-4 text-lg sm:text-xl leading-relaxed text-slate-300">
              하나는 <span className="text-white font-semibold">무엇을 계산할 것인가</span>의 문제이고,
              다른 하나는 <span className="text-white font-semibold">그 계산을 어디에 머물게 할 것인가</span>의 문제다.
              AICF는 이 두 층위를 분리해 설명하고, 다시 하나의 실행 구조로 연결하려는 시도다.
            </p>
          </div>
        </section>

        {/* Split View */}
        <section className="mt-10 grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left - Compute */}
          <div className="rounded-[2.5rem] border border-cyan-500/20 bg-[#111827] p-8 sm:p-10 shadow-xl">
            <div className="flex items-center gap-3 text-cyan-400 font-mono text-xs uppercase tracking-[0.3em] font-black">
              <BrainCircuit size={18} />
              Compute Optimization
            </div>

            <h2 className="mt-5 text-3xl sm:text-4xl font-black tracking-tight text-white leading-tight">
              의미를 유지하며
              <br />
              계산을 줄인다
            </h2>

            <p className="mt-6 text-slate-400 leading-relaxed text-base sm:text-lg">
              이 축은 모델 의미를 가능한 한 유지한 채, 연산량 자체를 줄이거나
              더 효율적인 수학적 형태로 재구성하는 문제를 다룬다.
            </p>

            <div className="mt-8 space-y-5">
              <div>
                <div className="text-slate-500 text-[11px] uppercase tracking-widest font-black">
                  Core Goal
                </div>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  compute reduction, approximation under constraints,
                  mathematically justified simplification
                </p>
              </div>

              <div>
                <div className="text-slate-500 text-[11px] uppercase tracking-widest font-black">
                  Main Methods
                </div>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  Quantization, Pruning, Distillation, Approximate Equivalence,
                  Constraint-aware Optimization
                </p>
              </div>

              <div>
                <div className="text-slate-500 text-[11px] uppercase tracking-widest font-black">
                  Philosophy
                </div>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  “조금 달라도 결과의 의미가 유지된다면 허용될 수 있다.”
                </p>
              </div>

              <div>
                <div className="text-slate-500 text-[11px] uppercase tracking-widest font-black">
                  AICF Role
                </div>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  모델의 수학적 그래프를 재구성하고, 의미 보존 조건과 최적화
                  제약을 정의한다.
                </p>
              </div>
            </div>
          </div>

          {/* Right - Memory */}
          <div className="rounded-[2.5rem] border border-emerald-500/20 bg-[#111827] p-8 sm:p-10 shadow-xl">
            <div className="flex items-center gap-3 text-emerald-400 font-mono text-xs uppercase tracking-[0.3em] font-black">
              <HardDrive size={18} />
              Memory Optimization
            </div>

            <h2 className="mt-5 text-3xl sm:text-4xl font-black tracking-tight text-white leading-tight">
              의미는 그대로 두고
              <br />
              이동을 줄인다
            </h2>

            <p className="mt-6 text-slate-400 leading-relaxed text-base sm:text-lg">
              이 축은 연산 자체보다, 연산 사이를 오가는 실제 데이터 이동을
              문제로 본다. 핵심은 HBM 왕복과 중간 결과 저장을 줄여 traffic
              병목을 제거하는 것이다.
            </p>

            <div className="mt-8 space-y-5">
              <div>
                <div className="text-slate-500 text-[11px] uppercase tracking-widest font-black">
                  Core Goal
                </div>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  traffic reduction, on-chip reuse, elimination of unnecessary
                  memory movement
                </p>
              </div>

              <div>
                <div className="text-slate-500 text-[11px] uppercase tracking-widest font-black">
                  Main Methods
                </div>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  Kernel Fusion, Tiling, Register Residency, Shared Memory
                  Reuse, Spill Avoidance
                </p>
              </div>

              <div>
                <div className="text-slate-500 text-[11px] uppercase tracking-widest font-black">
                  Philosophy
                </div>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  “연산은 정확해야 한다. 대신 메모리에는 가지 마.”
                </p>
              </div>

              <div>
                <div className="text-slate-500 text-[11px] uppercase tracking-widest font-black">
                  AICF Role
                </div>
                <p className="mt-1 text-slate-300 leading-relaxed">
                  하드웨어의 물리적 실행 경로를 재설계하고, register/shared
                  memory/HBM 사이의 데이터 흐름을 조직한다.
                </p>
              </div>
            </div>
          </div>
        </section>

        {/* Bottom Connection */}
        <section className="mt-10">
          <div className="rounded-[2.5rem] border border-slate-800 bg-gradient-to-br from-[#111827] to-[#0f172a] p-8 sm:p-10 shadow-xl">
            <div className="flex items-center gap-3 text-blue-400 font-mono text-xs uppercase tracking-[0.3em] font-black">
              <Workflow size={18} />
              AICF Position
            </div>

            <h2 className="mt-5 text-3xl sm:text-4xl font-black tracking-tight text-white leading-tight">
              AICF는 이 둘 중 하나만 고르는 구조가 아니다
            </h2>

            <p className="mt-6 text-slate-400 text-base sm:text-lg leading-relaxed max-w-5xl">
              AICF는 의미를 해석하는 계층과, 실제 물리적 데이터 이동을 조직하는
              계층을 분리해서 본다. 하나는{" "}
              <span className="text-cyan-400 font-semibold">Semantic Constraint</span>,
              다른 하나는{" "}
              <span className="text-emerald-400 font-semibold">Physical Dataflow</span>
              의 문제다. 이 둘을 실행 계획 계층으로 연결하는 것이 AICF의 구조적
              목표다.
            </p>

            <div className="mt-8 grid grid-cols-1 sm:grid-cols-3 gap-4">
              <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
                <div className="flex items-center gap-2 text-cyan-400 font-black">
                  <Sigma size={16} />
                  Semantic
                </div>
                <p className="mt-2 text-sm text-slate-400 leading-relaxed">
                  의미, 동일성, 허용 오차, 수학적 제약
                </p>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
                <div className="flex items-center gap-2 text-blue-400 font-black">
                  <Workflow size={16} />
                  Planner
                </div>
                <p className="mt-2 text-sm text-slate-400 leading-relaxed">
                  그래프 해석, 패턴 매칭, 실행 계획 구성
                </p>
              </div>

              <div className="rounded-2xl border border-slate-800 bg-slate-900/60 p-5">
                <div className="flex items-center gap-2 text-emerald-400 font-black">
                  <HardDrive size={16} />
                  Physical
                </div>
                <p className="mt-2 text-sm text-slate-400 leading-relaxed">
                  traffic, residency, reuse, memory path control
                </p>
              </div>
            </div>
          </div>
        </section>
      </main>
    </div>
  );
}