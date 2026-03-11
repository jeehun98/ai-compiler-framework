import React from "react";
import { Link } from "react-router-dom";
import { memoryPhaseLabels } from "../../data/memory/phaseLabels";

export default function MethodPhaseTabs({
  phaseSequence,
  activePhase,
  getPhaseLink,
}) {
  return (
    <div className="mb-10 overflow-x-auto">
      <div className="flex gap-2 p-1 bg-slate-900/40 w-max rounded-2xl border border-slate-800/50">
        {phaseSequence.map((phase) => (
          <Link
            key={phase}
            to={getPhaseLink(phase)}
            className={`px-5 py-2.5 rounded-xl text-xs font-black uppercase tracking-widest transition-all ${
              activePhase === phase
                ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20"
                : "text-slate-500 hover:text-slate-200 hover:bg-slate-800"
            }`}
          >
            {memoryPhaseLabels[phase]}
          </Link>
        ))}
      </div>
    </div>
  );
}