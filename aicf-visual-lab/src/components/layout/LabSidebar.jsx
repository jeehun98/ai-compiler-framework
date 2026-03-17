import React, { useState, useEffect, useMemo } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { allAnalysisConfigs } from "../../data/analysis/configs/index.js";

import {
  ArrowUpRight,
  ChevronRight,
  ChevronDown,
  Zap,
  FlaskConical,
  Activity,
  ShieldCheck,
  Workflow,
  Microscope,
  Beaker,
  Layers,
} from "lucide-react";

export default function LabSidebar({
  isOpen,
  onClose,
  version = "v1.0.0 Validation",
}) {
  const location = useLocation();
  const { opId, kernelId } = useParams();
  const pathname = location.pathname;

  const isLabHome = pathname === "/lab";
  const isPipeline = pathname === "/lab/pipeline" || pathname.startsWith("/lab/pipeline/");
  const isAnalysis = pathname === "/lab/analysis" || pathname.startsWith("/lab/analysis/");
  const isExperiments =
    pathname === "/lab/experiments" || pathname.startsWith("/lab/experiments/");

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
            ? "bg-violet-600/10 text-violet-400 border-violet-500/20"
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
            <div className="bg-violet-600 p-2 rounded-xl group-hover:bg-violet-500 transition shadow-lg shadow-violet-600/20">
              <FlaskConical size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-lg font-black tracking-tight text-white leading-none">
                AICF LAB
              </h1>
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                {version}
              </span>
            </div>
          </Link>
        </div>

        <nav className="p-4 space-y-1 border-b border-slate-800">
          <SectionTitle>Navigation</SectionTitle>
          {navItem("/lab", "Overview", Layers, { exact: true })}
          {navItem("/lab/pipeline", "Execution Pipeline", Workflow)}
          {navItem("/lab/analysis", "Kernel Analysis", Microscope)}
          {navItem("/lab/experiments", "Experiments", Beaker)}
        </nav>

        <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-slate-800">
          {isPipeline && (
            <>
              <SectionTitle>Execution Pipeline</SectionTitle>

              <div className="space-y-2">
                <Link
                  to="/lab/pipeline"
                  onClick={onClose}
                  className={[
                    "w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm border",
                    pathname === "/lab/pipeline"
                      ? "bg-violet-600/10 text-violet-400 border-violet-500/20"
                      : "text-slate-400 border-transparent hover:bg-slate-800 hover:text-slate-200",
                  ].join(" ")}
                >
                  <div className="flex flex-col items-start text-left">
                    <span className="tracking-tight flex items-center gap-2">
                      Overview
                      {pathname === "/lab/pipeline" && (
                        <Zap
                          size={10}
                          className="text-yellow-300 animate-pulse fill-yellow-300"
                        />
                      )}
                    </span>
                    <span className="text-[9px] mt-0.5 uppercase tracking-tighter font-black text-slate-500">
                      Traceable Execution Flow
                    </span>
                  </div>
                  <ChevronRight
                    size={14}
                    className={pathname === "/lab/pipeline" ? "text-violet-400" : "opacity-20"}
                  />
                </Link>
              </div>

              <div className="mt-4 rounded-2xl border border-slate-800 bg-[#111827] p-4">
                <p className="text-[10px] uppercase tracking-widest font-black text-slate-500">
                  Pipeline Focus
                </p>
                <div className="mt-3 space-y-2">
                  {[
                    "Capture Execution",
                    "Read Signals",
                    "Interpret Runtime State",
                    "Validate Assumptions",
                  ].map((step) => (
                    <div
                      key={step}
                      className="flex items-center gap-2 text-sm text-slate-400"
                    >
                      <div className="w-1.5 h-1.5 rounded-full bg-violet-500" />
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {isAnalysis && (
            <>
              <SectionTitle>Kernel Analysis</SectionTitle>

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
                          ? "bg-violet-600/10 text-violet-400 border-violet-500/20"
                          : "text-slate-400 border-transparent hover:bg-slate-800",
                      ].join(" ")}
                    >
                      <Link
                        to={`/lab/analysis/${id}`}
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
                            isOpActive ? "text-violet-200/80" : "text-slate-500"
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
                              to={`/lab/analysis/${id}/${k.id}`}
                              onClick={onClose}
                              className={[
                                "flex items-center justify-between px-3 py-2 rounded-lg text-[13px] font-bold transition-all",
                                isKernelActive
                                  ? "text-violet-400 bg-violet-400/5"
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

          {isExperiments && (
            <>
              <SectionTitle>Experiments</SectionTitle>

              <div className="space-y-2">
                {[
                  {
                    to: "/lab/experiments",
                    title: "Overview",
                    subtitle: "Validation and controlled testing",
                  },
                  {
                    to: "/lab/experiments/runtime",
                    title: "Runtime Conditions",
                    subtitle: "Input, shape, and device state",
                  },
                  {
                    to: "/lab/experiments/reports",
                    title: "Reports",
                    subtitle: "Measured outcomes and summaries",
                  },
                ].map((item) => {
                  const isActive = pathname === item.to;

                  return (
                    <Link
                      key={item.to}
                      to={item.to}
                      onClick={onClose}
                      className={[
                        "w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm border",
                        isActive
                          ? "bg-violet-600/10 text-violet-400 border-violet-500/20"
                          : "text-slate-400 border-transparent hover:bg-slate-800 hover:text-slate-200",
                      ].join(" ")}
                    >
                      <div className="flex flex-col items-start text-left">
                        <span className="tracking-tight flex items-center gap-2">
                          {item.title}
                          {isActive && (
                            <Zap
                              size={10}
                              className="text-yellow-300 animate-pulse fill-yellow-300"
                            />
                          )}
                        </span>
                        <span className="text-[9px] mt-0.5 uppercase tracking-tighter font-black text-slate-500">
                          {item.subtitle}
                        </span>
                      </div>
                      <ChevronRight
                        size={14}
                        className={isActive ? "text-violet-400" : "opacity-20"}
                      />
                    </Link>
                  );
                })}
              </div>

              <div className="mt-4 rounded-2xl border border-slate-800 bg-[#111827] p-4">
                <p className="text-[10px] uppercase tracking-widest font-black text-slate-500">
                  Experiment Focus
                </p>
                <div className="mt-3 space-y-2">
                  {[
                    "Controlled Inputs",
                    "Repeatable Conditions",
                    "Variant Comparison",
                    "Validation Reports",
                  ].map((step) => (
                    <div
                      key={step}
                      className="flex items-center gap-2 text-sm text-slate-400"
                    >
                      <div className="w-1.5 h-1.5 rounded-full bg-violet-500" />
                      <span>{step}</span>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {isLabHome && (
            <>
              <SectionTitle>Lab Overview</SectionTitle>

              <div className="rounded-2xl border border-slate-800 bg-[#111827] p-4 space-y-4">
                <div>
                  <p className="text-[10px] uppercase tracking-widest font-black text-slate-500">
                    Structure
                  </p>
                  <div className="mt-3 space-y-2">
                    {[
                      "Pipeline tracks execution flow",
                      "Analysis reads kernel behavior",
                      "Experiments validate design assumptions",
                      "Lab turns runtime into measurable signals",
                    ].map((line) => (
                      <div
                        key={line}
                        className="flex items-start gap-2 text-sm text-slate-400"
                      >
                        <div className="w-1.5 h-1.5 rounded-full bg-violet-500 mt-1.5" />
                        <span>{line}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="pt-2 border-t border-slate-800">
                  <p className="text-[10px] uppercase tracking-widest font-black text-slate-500">
                    Quick Links
                  </p>
                  <div className="mt-3 grid grid-cols-1 gap-2">
                    <Link
                      to="/lab/pipeline"
                      onClick={onClose}
                      className="px-3 py-2 rounded-xl text-sm font-bold text-slate-400 hover:bg-slate-800 hover:text-white border border-transparent"
                    >
                      Execution Pipeline
                    </Link>
                    <Link
                      to="/lab/analysis"
                      onClick={onClose}
                      className="px-3 py-2 rounded-xl text-sm font-bold text-slate-400 hover:bg-slate-800 hover:text-white border border-transparent"
                    >
                      Kernel Analysis
                    </Link>
                    <Link
                      to="/lab/experiments"
                      onClick={onClose}
                      className="px-3 py-2 rounded-xl text-sm font-bold text-slate-400 hover:bg-slate-800 hover:text-white border border-transparent"
                    >
                      Experiments
                    </Link>
                  </div>
                </div>
              </div>
            </>
          )}

          {!isLabHome && !isPipeline && !isAnalysis && !isExperiments && (
            <div className="px-3 py-10 text-center opacity-40">
              <Layers size={24} className="mx-auto mb-2" />
              <p className="text-[10px] uppercase font-black tracking-widest text-slate-500">
                No Active Lab Context
              </p>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-slate-800 bg-[#0b0f1a] text-[10px]">
          <div className="flex items-center gap-2 mb-2 text-violet-500">
            <ShieldCheck size={12} strokeWidth={3} />
            <span className="font-black uppercase tracking-widest">
              Measured Reality
            </span>
          </div>
          <p className="text-slate-600 font-medium leading-tight italic">
            "From runtime traces
            <br />
            to validated understanding."
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