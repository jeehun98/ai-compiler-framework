import React from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, ArrowRight } from "lucide-react";
import { memoryPhaseLabels } from "../../data/memory/phaseLabels";

export default function MethodPhasePager({
  prevPhase,
  nextPhase,
  getPhaseLink,
}) {
  return (
    <div className="mt-12 flex items-center justify-between border-t border-slate-800 pt-8">
      <div className="min-w-[160px]">
        {prevPhase && (
          <Link
            to={getPhaseLink(prevPhase)}
            className="group flex flex-col items-start gap-1"
          >
            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
              Previous Phase
            </span>
            <div className="inline-flex items-center gap-2 text-slate-300 font-bold group-hover:text-white transition">
              <ArrowLeft size={16} className="group-hover:-translate-x-1 transition-transform" />
              {memoryPhaseLabels[prevPhase]}
            </div>
          </Link>
        )}
      </div>

      <div className="min-w-[160px] text-right">
        {nextPhase && (
          <Link
            to={getPhaseLink(nextPhase)}
            className="group flex flex-col items-end gap-1"
          >
            <span className="text-[10px] font-black text-slate-500 uppercase tracking-widest">
              Next Phase
            </span>
            <div className="inline-flex items-center gap-2 text-emerald-400 font-bold group-hover:text-emerald-300 transition">
              {memoryPhaseLabels[nextPhase]}
              <ArrowRight size={16} className="group-hover:translate-x-1 transition-transform" />
            </div>
          </Link>
        )}
      </div>
    </div>
  );
}