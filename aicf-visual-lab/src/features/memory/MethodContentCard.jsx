import React from "react";

export default function MethodContentCard({ content, Icon }) {
  return (
    <section className="rounded-[2.5rem] border border-slate-800 bg-[#111827] p-8 md:p-12 space-y-10 shadow-2xl relative overflow-hidden">
      <div className="absolute top-0 right-0 p-12 opacity-[0.02] pointer-events-none">
        <Icon size={300} />
      </div>

      <div className="relative z-10">
        <h2 className="text-2xl md:text-3xl font-black text-white mb-6 flex items-center gap-3">
          <span className="w-8 h-[2px] bg-emerald-500"></span>
          {content.title}
        </h2>

        {content.summary && (
          <p className="text-slate-300 text-lg leading-relaxed font-light mb-10 italic border-l-4 border-emerald-500/30 pl-6">
            {content.summary}
          </p>
        )}

        <div className="grid grid-cols-1 gap-10">
          {content.problem && (
            <div className="bg-slate-900/50 p-6 rounded-2xl border border-slate-800">
              <h3 className="text-emerald-400 text-[10px] font-black uppercase tracking-widest mb-3">
                The Problem
              </h3>
              <p className="text-slate-400 leading-relaxed">{content.problem}</p>
            </div>
          )}

          <div className="space-y-8">
            {content.property && (
              <div>
                <h3 className="text-white text-lg font-black mb-3">Key Mechanism</h3>
                <p className="text-slate-400 leading-relaxed">{content.property}</p>
              </div>
            )}

            {content.impact && (
              <div>
                <h3 className="text-white text-lg font-black mb-3">Architectural Impact</h3>
                <p className="text-slate-400 leading-relaxed">{content.impact}</p>
              </div>
            )}

            {content.body?.length > 0 && (
              <div className="space-y-4 pt-4 border-t border-slate-800">
                {content.body.map((paragraph, idx) => (
                  <p key={idx} className="text-slate-400 leading-relaxed font-light">
                    {paragraph}
                  </p>
                ))}
              </div>
            )}
          </div>
        </div>

        {content.bullets?.length > 0 && (
          <div className="flex flex-wrap gap-2 mt-10">
            {content.bullets.map((item) => (
              <span
                key={item}
                className="px-4 py-1.5 rounded-full bg-emerald-500/5 text-[10px] font-black text-emerald-400/80 border border-emerald-500/10"
              >
                {item}
              </span>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}