import React, { useState, useEffect, useMemo } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { allAnalysisConfigs } from "../../data/analysis/configs/index.js";
import { allOpsData } from "../../data/ops/index.js";
import { theoryByOpId } from "../../data/theory/index.js";

import {
  Terminal,
  BookOpen,
  ArrowUpRight,
  ChevronRight,
  ChevronDown,
  Zap,
  Beaker,
  Layers,
  ShieldCheck,
  FlaskConical,
  Workflow,
} from "lucide-react";

export default function ComputeSidebar({
  isOpen,
  onClose,
  version = "v1.1.0 Semantic",
}) {
  const location = useLocation();
  const { opId, kernelId } = useParams();
  const pathname = location.pathname;

  const searchParams = new URLSearchParams(location.search);
  const selectedOpId = searchParams.get("op");

  const isComputeHome = pathname === "/compute";
  const isTheory = pathname === "/compute/theory" || pathname.startsWith("/compute/theory/");
  const isOps = pathname === "/compute/ops" || pathname.startsWith("/compute/ops/");
  const isAnalysis =
    pathname === "/compute/analysis" || pathname.startsWith("/compute/analysis/");
  const isPipeline =
    pathname === "/compute/pipeline" || pathname.startsWith("/compute/pipeline/");

  const analysisData = useMemo(
    () =>
      Object.fromEntries(
        Object.entries(allAnalysisConfigs).map(([op, cfg]) => [
          op,
          {
            label: cfg.label,
            category: cfg.category,
            kernels: cfg.variants.map((v) => ({
              id: v.id,
              label: v.name,
              tag: v.tag,
            })),
          },
        ])
      ),
    []
  );

  const analysisOpIds = useMemo(() => Object.keys(analysisData), [analysisData]);
  const firstAnalysisOp = analysisOpIds[0] || null;

  const [expandedOp, setExpandedOp] = useState(opId || firstAnalysisOp);

  useEffect(() => {
    if (isAnalysis) {
      if (opId) {
        setExpandedOp(opId);
      } else if (!expandedOp && firstAnalysisOp) {
        setExpandedOp(firstAnalysisOp);
      }
    }
  }, [isAnalysis, opId, expandedOp, firstAnalysisOp]);

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
            to="/compute"
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
          {navItem("/compute/theory", "Theory Specs", BookOpen)}
          {navItem("/compute/ops", "Ops Explorer", Terminal)}
          {navItem("/compute/analysis", "Kernel Analysis Lab", FlaskConical)}
          {navItem("/compute/pipeline", "Execution Pipeline", Workflow)}
        </nav>

        <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-slate-800">
          {isTheory && (
            <>
              <SectionTitle>Theory Specifications</SectionTitle>
              {Object.keys(theoryByOpId).map((id) => {
                const isSelected = selectedOpId === id;

                return (
                  <Link
                    key={id}
                    to={`/compute/theory?op=${id}`}
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
                        {theoryByOpId[id]?.subtitle || "Spec"}
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

          {isAnalysis && (
            <>
              <SectionTitle>Kernel Analysis Lab</SectionTitle>

              {analysisOpIds.map((id) => {
                const op = analysisData[id];
                const isExpanded = expandedOp === id;
                const isOpActive = opId === id && !kernelId;

                return (
                  <div key={id} className="space-y-1">
                    <div
                      className={[
                        "w-full flex items-center justify-between rounded-xl transition-all font-bold text-sm border",
                        isOpActive
                          ? "bg-emerald-600/10 text-emerald-400 border-emerald-500/20"
                          : "text-slate-400 border-transparent hover:bg-slate-800",
                      ].join(" ")}
                    >
                      <Link
                        to={`/compute/analysis/${id}`}
                        className="flex-1 text-left flex flex-col px-4 py-3"
                        onClick={onClose}
                      >
                        <span className="flex items-center gap-2">
                          {op.label}
                          {isOpActive && (
                            <Zap
                              size={10}
                              className="text-yellow-300 animate-pulse fill-yellow-300"
                            />
                          )}
                        </span>
                        <span
                          className={`text-[9px] uppercase font-black ${
                            isOpActive ? "text-emerald-200/80" : "text-slate-500"
                          }`}
                        >
                          {op.category}
                        </span>
                      </Link>

                      <button
                        onClick={() => setExpandedOp(isExpanded ? null : id)}
                        className="p-3 hover:bg-white/10 rounded-r-xl"
                        type="button"
                        aria-label={isExpanded ? "Collapse kernels" : "Expand kernels"}
                      >
                        {isExpanded ? (
                          <ChevronDown size={14} />
                        ) : (
                          <ChevronRight size={14} />
                        )}
                      </button>
                    </div>

                    {isExpanded && (
                      <div className="ml-4 pl-4 border-l border-slate-800 space-y-1 mt-1 animate-in slide-in-from-top-2 duration-200">
                        {op.kernels.map((k) => {
                          const isKernelActive = opId === id && kernelId === k.id;

                          return (
                            <Link
                              key={k.id}
                              to={`/compute/analysis/${id}/${k.id}`}
                              onClick={onClose}
                              className={[
                                "flex items-center justify-between px-3 py-2 rounded-lg text-[13px] font-bold transition-all",
                                isKernelActive
                                  ? "text-emerald-400 bg-emerald-400/5"
                                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50",
                              ].join(" ")}
                            >
                              <span className="flex items-center gap-2 min-w-0">
                                <Zap
                                  size={12}
                                  className={
                                    isKernelActive
                                      ? "text-yellow-400 fill-yellow-400"
                                      : "text-slate-600"
                                  }
                                />
                                <span className="truncate">{k.label}</span>
                              </span>

                              <div className="flex items-center gap-2 shrink-0">
                                <span className="text-[8px] border border-slate-700 px-1.5 py-0.5 rounded uppercase tracking-tighter opacity-60">
                                  {k.tag}
                                </span>
                                {isKernelActive && <ArrowUpRight size={12} />}
                              </div>
                            </Link>
                          );
                        })}
                      </div>
                    )}
                  </div>
                );
              })}
            </>
          )}

          {isPipeline && (
            <>
              <SectionTitle>Execution Pipeline</SectionTitle>

              <div className="space-y-2">
                <Link
                  to="/compute/pipeline"
                  onClick={onClose}
                  className={[
                    "w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm border",
                    pathname === "/compute/pipeline"
                      ? "bg-emerald-600/10 text-emerald-400 border-emerald-500/20"
                      : "text-slate-400 border-transparent hover:bg-slate-800 hover:text-slate-200",
                  ].join(" ")}
                >
                  <div className="flex flex-col items-start text-left">
                    <span className="tracking-tight flex items-center gap-2">
                      Overview
                      {pathname === "/compute/pipeline" && (
                        <Zap
                          size={10}
                          className="text-yellow-300 animate-pulse fill-yellow-300"
                        />
                      )}
                    </span>
                    <span className="text-[9px] mt-0.5 uppercase tracking-tighter font-black text-slate-500">
                      Dynamic Execution Flow
                    </span>
                  </div>
                  <ChevronRight
                    size={14}
                    className={pathname === "/compute/pipeline" ? "text-emerald-400" : "opacity-20"}
                  />
                </Link>
              </div>

              <div className="mt-4 rounded-2xl border border-slate-800 bg-[#111827] p-4">
                <p className="text-[10px] uppercase tracking-widest font-black text-slate-500">
                  Pipeline Focus
                </p>
                <div className="mt-3 space-y-2">
                  {[
                    "Invariant Check",
                    "Path Search",
                    "Parameter Binding",
                    "Kernel Realization",
                  ].map((step) => (
                    <div
                      key={step}
                      className="flex items-center gap-2 text-sm text-slate-400"
                    >
                      <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}


          {!isComputeHome &&
            !isTheory &&
            !isOps &&
            !isAnalysis &&
            !isPipeline && (
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
            "From invariant meaning
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