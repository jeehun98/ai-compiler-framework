import React from "react";
import { Link, useLocation } from "react-router-dom";
import { allOpsData } from "../../data/ops/index.js";
import { theoryByPropertyId } from "../../data/theory/index.js";

import {
  Terminal,
  BookOpen,
  ChevronRight,
  Zap,
  Beaker,
  Layers,
  ShieldCheck,
} from "lucide-react";

export default function ComputeSidebar({
  isOpen,
  onClose,
  version = "v1.1.0 Semantic",
}) {
  const location = useLocation();
  const pathname = location.pathname;

  const searchParams = new URLSearchParams(location.search);
  const selectedOpId = searchParams.get("op");
  const selectedTheoryPropertyId = searchParams.get("property");

  const isComputeHome = pathname === "/compute";
  const isTheory =
    pathname === "/compute/theory" || pathname.startsWith("/compute/theory/");
  const isOps =
    pathname === "/compute/ops" || pathname.startsWith("/compute/ops/");

  const SectionTitle = ({ children }) => (
    <p className="px-3 text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2 mt-4 first:mt-0">
      {children}
    </p>
  );

  const navItem = (to, label, Icon, options = {}) => {
    const { exact = false } = options;

    const isActive = exact
      ? pathname === to
      : pathname === to || pathname.startsWith(`${to}/`);

    return (
      <Link
        to={to}
        onClick={onClose}
        className={[
          "flex items-center gap-3 px-3 py-2.5 rounded-xl transition font-bold text-sm border",
          isActive
            ? "bg-emerald-600/10 text-emerald-400 border-emerald-500/20"
            : "text-slate-400 border-transparent hover:bg-slate-800 hover:text-white",
        ].join(" ")}
      >
        <Icon size={18} />
        {label}
      </Link>
    );
  };

  return (
    <>
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 md:hidden"
          onClick={onClose}
        />
      )}

      <aside
        className={[
          "fixed md:static inset-y-0 left-0 z-50 md:z-10 w-[85vw] max-w-[320px] md:w-80",
          "bg-[#0f172a] border-r border-slate-800 flex flex-col shadow-2xl transition-transform duration-300",
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        ].join(" ")}
      >
        <div className="p-6 border-b border-slate-800 bg-[#0b0f1a]">
          <Link
            to="/"
            className="flex items-center gap-3 group"
            onClick={onClose}
          >
            <div className="bg-emerald-600 p-2 rounded-xl group-hover:bg-emerald-500 transition shadow-lg shadow-emerald-600/20">
              <Beaker size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-lg font-black tracking-tight text-white leading-none">
                AICF COMPUTE
              </h1>
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                {version}
              </span>
            </div>
          </Link>
        </div>

        <nav className="p-4 space-y-1 border-b border-slate-800">
          <SectionTitle>Navigation</SectionTitle>
          {navItem("/compute", "Overview", Layers, { exact: true })}
          {navItem("/compute/theory", "Property Atlas", BookOpen)}
          {navItem("/compute/ops", "Ops Explorer", Terminal)}
        </nav>

        <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-slate-800">
          {isTheory && (
            <>
              <SectionTitle>Compute Properties</SectionTitle>

              {Object.keys(theoryByPropertyId).map((id) => {
                const isSelected = selectedTheoryPropertyId === id;
                const spec = theoryByPropertyId[id];

                return (
                  <Link
                    key={id}
                    to={`/compute/theory?property=${id}`}
                    onClick={onClose}
                    className={[
                      "w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm mb-1 border",
                      isSelected
                        ? "bg-emerald-600/10 text-emerald-400 border-emerald-500/20"
                        : "text-slate-400 border-transparent hover:bg-slate-800 hover:text-slate-200",
                    ].join(" ")}
                  >
                    <div className="min-w-0 flex flex-col items-start text-left">
                      <span className="truncate tracking-tight flex items-center gap-2">
                        {id}
                        {isSelected && (
                          <Zap
                            size={10}
                            className="text-yellow-300 animate-pulse fill-yellow-300"
                          />
                        )}
                      </span>

                      <span
                        className={`text-[9px] mt-0.5 uppercase tracking-tighter font-black ${
                          isSelected ? "text-emerald-200/80" : "text-slate-500"
                        }`}
                      >
                        {spec?.subtitle || "Property"}
                      </span>
                    </div>

                    <ChevronRight
                      size={14}
                      className={isSelected ? "text-emerald-400" : "opacity-20"}
                    />
                  </Link>
                );
              })}
            </>
          )}

          {isOps && (
            <>
              <SectionTitle>Operator Library</SectionTitle>

              {Object.keys(allOpsData).map((id) => {
                const isSelected = selectedOpId === id;

                return (
                  <Link
                    key={id}
                    to={`/compute/ops?op=${id}`}
                    onClick={onClose}
                    className={[
                      "w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm mb-1 border",
                      isSelected
                        ? "bg-emerald-600/10 text-emerald-400 border-emerald-500/20"
                        : "text-slate-400 border-transparent hover:bg-slate-800 hover:text-slate-200",
                    ].join(" ")}
                  >
                    <div className="min-w-0 flex flex-col items-start text-left">
                      <span className="truncate tracking-tight flex items-center gap-2">
                        {id}
                        {isSelected && (
                          <Zap
                            size={10}
                            className="text-yellow-300 animate-pulse fill-yellow-300"
                          />
                        )}
                      </span>

                      <span
                        className={`text-[9px] mt-0.5 uppercase tracking-tighter font-black ${
                          isSelected ? "text-emerald-200/80" : "text-slate-500"
                        }`}
                      >
                        {allOpsData[id]?.category || "Uncategorized"}
                      </span>
                    </div>

                    <ChevronRight
                      size={14}
                      className={isSelected ? "text-emerald-400" : "opacity-20"}
                    />
                  </Link>
                );
              })}
            </>
          )}

          {isComputeHome && (
            <>
              <SectionTitle>Compute Overview</SectionTitle>

              <div className="rounded-2xl border border-slate-800 bg-[#111827] p-4 space-y-4">
                <div>
                  <p className="text-[10px] uppercase tracking-widest font-black text-slate-500">
                    Structure
                  </p>

                  <div className="mt-3 space-y-2">
                    {[
                      "Theory defines semantic properties",
                      "Ops maps operators to property profiles",
                      "Compute focuses on invariant-preserving execution",
                    ].map((line) => (
                      <div
                        key={line}
                        className="flex items-start gap-2 text-sm text-slate-400"
                      >
                        <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 mt-1.5" />
                        <span>{line}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </>
          )}

          {!isComputeHome && !isTheory && !isOps && (
            <div className="px-3 py-10 text-center opacity-40">
              <Layers size={24} className="mx-auto mb-2" />
              <p className="text-[10px] uppercase font-black tracking-widest text-slate-500">
                No Active Compute Context
              </p>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-slate-800 bg-[#0b0f1a] text-[10px]">
          <div className="flex items-center gap-2 mb-2 text-emerald-500">
            <ShieldCheck size={12} strokeWidth={3} />
            <span className="font-black uppercase tracking-widest">
              Semantic Boundary
            </span>
          </div>

          <p className="text-slate-600 font-medium leading-tight italic">
            "From semantic property
            <br />
            to executable form."
          </p>
        </div>
      </aside>

      <style jsx="true">{`
        .scrollbar-thin::-webkit-scrollbar {
          width: 4px;
        }
        .scrollbar-thin::-webkit-scrollbar-thumb {
          background: #1f2937;
          border-radius: 10px;
        }
      `}</style>
    </>
  );
}