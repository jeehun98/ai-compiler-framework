// src/pages/HomePage.jsx
import React from "react";
import { Link } from "react-router-dom";
import { Cpu, Zap, ShieldCheck, Layers, ArrowRight, Boxes } from "lucide-react";

export default function HomePage() {
  return (
    <div className="min-h-screen bg-[#0f172a] text-slate-200">
      <div className="max-w-7xl mx-auto px-8 py-14 space-y-16">
        {/* HERO */}
        <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-12 shadow-2xl relative overflow-hidden">
          <div className="absolute -top-10 -right-10 text-[160px] font-black text-blue-500/5 italic pointer-events-none">
            AICF
          </div>

          <div className="flex items-center gap-2 text-blue-400 font-mono text-xs uppercase tracking-[0.35em] font-black">
            <Cpu size={16} /> Semantic Compiler Framework
          </div>

          <h1 className="mt-6 text-6xl font-black tracking-tighter italic leading-[0.95]">
            의미론을 고정한 채,
            <br />
            커널로 내리는 컴파일러
          </h1>

          <p className="mt-6 max-w-3xl text-slate-400 leading-relaxed italic">
            OpFlags / Invariants로 “의미 계약”을 먼저 박고, 그 제약 안에서만 Fusion·Lowering·Kernel Variant 선택이 일어납니다.
          </p>

          <div className="mt-10 flex flex-wrap gap-3">
            <Link
              to="/ops"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-2xl bg-blue-600 text-white font-black text-xs uppercase tracking-widest shadow-lg hover:opacity-90 transition"
            >
              연산 의미론 보기 <ArrowRight size={16} />
            </Link>

            {/* 앵커는 a 유지 */}
            <a
              href="#architecture"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-2xl border border-slate-700 text-slate-200 font-black text-xs uppercase tracking-widest hover:bg-slate-800 transition"
            >
              구성 보기 <ArrowRight size={16} />
            </a>
          </div>

          <div className="mt-10 flex items-center gap-2 text-emerald-400 font-black bg-emerald-400/5 px-4 py-2 rounded-xl border border-emerald-400/10 text-xs uppercase tracking-widest w-fit">
            <ShieldCheck size={16} /> Semantic Anchored
          </div>
        </section>

        {/* WHY */}
        <section className="grid grid-cols-12 gap-6">
          <div className="col-span-12 lg:col-span-7 bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl">
            <div className="flex items-center gap-2 text-purple-400 font-black uppercase tracking-widest text-xs">
              <Layers size={16} /> Why Semantics
            </div>
            <h2 className="mt-4 text-4xl font-black tracking-tighter italic">
              최적화가 의미를 깨뜨리는 순간을
              <br />
              시스템이 직접 관리한다
            </h2>
            <p className="mt-6 text-slate-400 leading-relaxed italic">
              성능을 위해 연산을 합치고 바꾸는 과정에서, 원래의 의미 제약은 암묵적으로 처리되기 쉽습니다. AICF는 의미 제약을 명시화하고, 그 제약을 만족하는 최적화만 허용합니다.
            </p>
          </div>

          <div className="col-span-12 lg:col-span-5 space-y-4">
            {[
              { title: "의미 계약(Invariants) 1st", desc: "허용되는 전략을 ‘의미’로 제한" },
              { title: "IR → Kernel 분리", desc: "의미 판단과 실행 전략을 분리" },
              { title: "증거 기반(Tests/Perf)", desc: "정합성과 성능을 함께 제시" },
            ].map((x) => (
              <div
                key={x.title}
                className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-8 shadow-xl hover:border-blue-500/30 transition"
              >
                <div className="text-blue-400 font-black uppercase tracking-widest text-xs">
                  {x.title}
                </div>
                <div className="mt-3 text-slate-400 italic">{x.desc}</div>
              </div>
            ))}
          </div>
        </section>

        {/* ARCHITECTURE MAP */}
        <section id="architecture" className="space-y-6">
          <div className="flex items-center gap-2 text-blue-400 font-black uppercase tracking-widest text-xs">
            <Boxes size={16} /> System Map
          </div>
          <h2 className="text-4xl font-black tracking-tighter italic">
            Meaning → IR → Kernel
          </h2>

          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {[
              {
                title: "Meaning (Contract)",
                icon: <ShieldCheck size={18} />,
                desc: "Op 정의, 축 의미, 불변성, 허용 전략(Allow List)",
              },
              {
                title: "IR (Transform)",
                icon: <Layers size={18} />,
                desc: "Fuse / Rewrite / Lowering Decision, 패스별 Diff",
              },
              {
                title: "Kernel (Realization)",
                icon: <Zap size={18} />,
                desc: "KID/Variant 선택, 런치 파라미터, 메모리 스토리, Perf",
              },
            ].map((c) => (
              <div
                key={c.title}
                className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl"
              >
                <div className="flex items-center gap-2 text-slate-400 font-mono text-xs uppercase tracking-[0.2em] font-black">
                  {c.icon} {c.title}
                </div>
                <p className="mt-6 text-slate-300 font-black italic text-xl tracking-tight">
                  {c.desc}
                </p>
              </div>
            ))}
          </div>
        </section>

        {/* DEMO ENTRY */}
        <section className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-10 shadow-xl">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <div className="text-slate-500 font-mono text-xs uppercase tracking-[0.35em] font-black">
                Demo Entry
              </div>
              <h3 className="mt-3 text-3xl font-black tracking-tighter italic">
                연산 의미론 DeepDive로 바로 들어가기
              </h3>
            </div>

            <Link
              to="/ops"
              className="inline-flex items-center gap-2 px-6 py-3 rounded-2xl bg-emerald-600/10 hover:bg-emerald-600 border border-emerald-500/30 text-emerald-400 hover:text-white font-black text-xs uppercase tracking-widest transition shadow-lg"
            >
              Open Ops Lab <ArrowRight size={16} />
            </Link>
          </div>

          <div className="mt-8 flex flex-wrap gap-2">
            {["AdamStep", "GemmEpilogue", "LayerNorm", "Softmax"].map((id) => (
              <Link
                key={id}
                to={`/ops?op=${id}`}
                className="text-[10px] font-black bg-slate-900 text-slate-400 px-3 py-2 rounded-xl border border-slate-800 uppercase tracking-widest hover:text-blue-300 transition"
              >
                {id}
              </Link>
            ))}
          </div>
        </section>

        {/* ROADMAP */}
        <section className="pb-10">
          <div className="text-slate-500 font-mono text-xs uppercase tracking-[0.35em] font-black">
            Roadmap
          </div>
          <div className="mt-4 grid grid-cols-1 lg:grid-cols-3 gap-6">
            {[
              { t: "IR Explorer", d: "패스 슬라이더 + 그래프 + Diff" },
              { t: "Kernel Explorer", d: "KID/Variant/Perf/Correctness" },
              { t: "Runtime Adaptation", d: "관찰 기반 정책/선택 로그 시각화" },
            ].map((x) => (
              <div
                key={x.t}
                className="bg-[#1e293b] border border-slate-800 rounded-[2.5rem] p-8 shadow-xl"
              >
                <div className="text-blue-400 font-black uppercase tracking-widest text-xs">
                  {x.t}
                </div>
                <div className="mt-3 text-slate-400 italic">{x.d}</div>
              </div>
            ))}
          </div>
        </section>
      </div>
    </div>
  );
}