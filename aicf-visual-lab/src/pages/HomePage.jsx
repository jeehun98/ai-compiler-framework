// src/pages/HomePage.jsx
import React from "react";
import { Link } from "react-router-dom";
import {
  Cpu,
  ShieldCheck,
  Layers,
  ArrowRight,
  Boxes,
  Sparkles,
  Waypoints,
} from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200 antialiased">
      <div className="max-w-7xl mx-auto px-8 py-14 space-y-16">
        
        {/* HERO SECTION */}
        <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-12 shadow-2xl relative overflow-hidden">
          {/* 배경 장식: 이탤릭을 제거하여 더 구조적인 느낌 전달 */}
          <div className="absolute -top-10 -right-10 text-[160px] font-black text-blue-500/5 pointer-events-none tracking-tighter">
            AICF
          </div>

          <div className="flex items-center gap-2 text-blue-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
            <Cpu size={16} /> AI Compiler Framework
          </div>

          {/* 메인 타이틀: 정체(Upright)와 두꺼운 굵기로 가독성 극대화 */}
          <h1 className="mt-6 text-6xl font-black tracking-tight leading-[1.1] text-white">
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
              <a
                href="#architecture"
                className="inline-flex items-center gap-2 px-6 py-4 rounded-2xl border border-slate-700 text-slate-300 font-bold text-xs uppercase tracking-widest hover:bg-slate-800 transition"
              >
                구조 보기
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
                p2: "연산의 순서와 데이터 흐름은 하나의 논리로서 계산 그래프를 이룬다.",
              },
              {
                k: "Step 2",
                t: "최적화의 역설",
                sub: "The Reality",
                icon: <Layers size={18} />,
                p1: "성능 최적화 레이어를 거치며, 수학적 구조는 하드웨어 친화적 형태로 치환된다.",
                p2: "Reorder/Fusion 과정에서 연산의 경계와 의미 단위가 흐려지기 쉽다.",
              },
              {
                k: "Step 3",
                t: "연산의 재정의",
                sub: "The Approach",
                icon: <ShieldCheck size={18} />,
                p1: "문제는 실행이 동적이라는 사실이 아니라, 의미가 구조적으로 정의되지 않았다는 점이다.",
                p2: "연산을 의미론적 객체로 정의하고, 그 의미를 데이터 구조로 고정한다.",
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
                  <div className="text-blue-100 font-black text-xl tracking-tight leading-tight">
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

          <div className="bg-[#0b1220] border border-blue-900/30 rounded-[2.5rem] p-8 shadow-xl">
            <div className="text-slate-500 font-mono text-xs uppercase tracking-[0.35em] font-black">
              Key idea
            </div>
            <p className="mt-3 text-slate-200 font-bold text-2xl tracking-tight">
              의미를 먼저 정의하고,{" "}
              <span className="text-blue-400">그 의미를 보존하는 방식으로</span>{" "}
              최적화를 설계한다.
            </p>
          </div>
        </section>

        {/* WHY SECTION */}
        <section className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-7 bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl">
            <div className="flex items-center gap-2 text-purple-400 font-black uppercase tracking-widest text-xs">
              <Layers size={16} /> Why Semantics
            </div>
            <h2 className="mt-4 text-4xl font-black tracking-tight text-white leading-tight">
              의미가 사라지는 지점을
              <br />
              구조적으로 다룬다
            </h2>
            <p className="mt-6 text-slate-400 leading-relaxed text-lg">
              최적화는 필수적입니다. 다만 그 과정에서 의미의 경계가 흐려질 때,
              어떤 변형이 허용되는지 설명하기 어려워집니다. 
              AICF는 <span className="text-blue-300">연산의 성질(Semantics)</span>을 
              제약 조건으로 삼아 안전한 변형만을 허용합니다.
            </p>
          </div>

          <div className="col-span-12 lg:col-span-5 space-y-4">
            {[
              { title: "의미 기반 정의", desc: "연산을 성질의 조합으로 표현" },
              { title: "의미 보존형 변형", desc: "허용되는 Rewrite를 규칙으로 제한" },
              { title: "증거 제시", desc: "정합성 테스트와 성능 근거 제공" },
            ].map((x) => (
              <div
                key={x.title}
                className="bg-[#1e293b] border border-slate-800 rounded-[2rem] p-7 shadow-xl hover:border-blue-500/30 transition group"
              >
                <div className="text-blue-400 font-black uppercase tracking-widest text-[10px]">
                  {x.title}
                </div>
                <div className="mt-2 text-slate-300 font-semibold group-hover:text-white transition">
                  {x.desc}
                </div>
              </div>
            ))}
          </div>
        </section>

        {/* ARCHITECTURE MAP */}
        <section id="architecture" className="space-y-8">
          <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
            <Boxes size={16} /> System Map
          </div>
          <h2 className="text-4xl font-black tracking-tight text-white">
            Meaning → IR → Kernel
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {[
              {
                title: "Meaning (Contract)",
                icon: <ShieldCheck size={18} />,
                desc: "연산 정의(성질/축/불변성)와 허용 변형 규칙",
              },
              {
                title: "IR (Transform)",
                icon: <Layers size={18} />,
                desc: "의미 보존 조건 하에서 Fuse / Rewrite 수행",
              },
              {
                title: "Kernel (Realization)",
                icon: <Cpu size={18} />,
                desc: "의미 계약을 실현하는 최적 커널 구성 및 분석",
              },
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

        {/* DEMO ENTRY */}
        <section className="bg-blue-600/5 border border-blue-500/20 rounded-[2.5rem] p-10 shadow-xl">
          <div className="flex items-center justify-between flex-wrap gap-8">
            <div className="max-w-xl">
              <div className="text-blue-500 font-mono text-xs uppercase tracking-[0.35em] font-black">
                Demo Entry
              </div>
              <h3 className="mt-3 text-3xl font-black tracking-tight text-white">
                연산 정의(의미)에서 시작하기
              </h3>
              <p className="mt-3 text-slate-400">
                Op 정의, 의미 제약, 허용 변형 규칙을 연산 단위로 직접 탐색하고 실험해 보세요.
              </p>
            </div>

            <Link
              to="/ops"
              className="inline-flex items-center gap-2 px-8 py-4 rounded-2xl bg-emerald-600 text-white font-bold text-sm uppercase tracking-widest transition shadow-lg hover:bg-emerald-500"
            >
              Open Ops Lab <ArrowRight size={18} />
            </Link>
          </div>

          <div className="mt-10 flex flex-wrap gap-2">
            {["AdamStep", "GemmEpilogue", "LayerNorm", "Softmax"].map((id) => (
              <Link
                key={id}
                to={`/ops?op=${id}`}
                className="text-[10px] font-bold bg-slate-900 text-slate-400 px-4 py-2 rounded-xl border border-slate-800 uppercase tracking-widest hover:text-blue-400 hover:border-blue-400/30 transition"
              >
                {id}
              </Link>
            ))}
          </div>
        </section>

        {/* ROADMAP */}
        <section className="pb-20">
          <div className="text-slate-500 font-mono text-xs uppercase tracking-[0.35em] font-black">
            Roadmap
          </div>
          <div className="mt-6 grid grid-cols-1 lg:grid-cols-3 gap-6 opacity-80">
            {[
              { t: "IR Explorer", d: "패스 흐름 + Diff로 의미 보존 시각화" },
              { t: "Rulebook Fusion", d: "의미 기반 허용 규칙 중심의 합성 엔진" },
              { t: "Kernel Explorer", d: "의미 계약 → 실현 커널 매핑 분석" },
            ].map((x) => (
              <div
                key={x.t}
                className="bg-[#0f172a] border border-slate-800 rounded-[2rem] p-8 shadow-sm"
              >
                <div className="text-blue-400/70 font-black uppercase tracking-widest text-[10px]">
                  {x.t}
                </div>
                <div className="mt-3 text-slate-500 text-sm font-medium">{x.d}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}