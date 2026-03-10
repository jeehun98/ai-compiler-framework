import React, { useState, useEffect } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import { allAnalysisConfigs } from "../data/analysis/configs";
import { allOpsData } from "../data/index.js";
import { theoryByOpId } from "../data/theory/index.js";

import {
  Terminal,
  BookOpen,
  ArrowUpRight,
  ChevronRight,
  ChevronDown,
  Microscope,
  Zap,
  Beaker,
  Layers,
  GitMerge,
  ShieldCheck,
} from "lucide-react";

export default function ComputeSidebar({
  isOpen,
  onClose,
  version = "v1.0.6 Lab-Ready",
}) {
  const location = useLocation();
  const { opId, kernelId } = useParams();
  const pathname = location.pathname;

  const isComputeHome = pathname === "/compute";
  const isOps = pathname.startsWith("/compute/ops");
  const isTheory = pathname.startsWith("/compute/theory");
  const isPipeline = pathname.startsWith("/compute/pipeline");
  const isAnalysis = pathname.startsWith("/compute/analysis");

  const analysisData = Object.fromEntries(
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
  );

  const [expandedOp, setExpandedOp] = useState(opId || "add");

  useEffect(() => {
    if (opId) setExpandedOp(opId);
  }, [opId]);

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
          "flex items-center gap-3 px-3 py-2.5 rounded-xl transition font-bold text-sm",
          isActive
            ? "bg-emerald-600/10 text-emerald-400 border border-emerald-500/20"
            : "text-slate-400 hover:bg-slate-800 hover:text-white",
        ].join(" ")}
      >
        <Icon size={18} /> {label}
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
        {/* Header */}
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

        {/* Top Navigation */}
        <nav className="p-4 space-y-1 border-b border-slate-800">
          <SectionTitle>Navigation</SectionTitle>
          {navItem("/compute", "Overview", Layers, { exact: true })}
          {navItem("/compute/theory", "Theory Specs", BookOpen)}
          {navItem("/compute/pipeline", "Compiler Pipeline", GitMerge)}
          {navItem("/compute/ops", "Ops Explorer", Terminal)}
          {navItem("/compute/analysis", "Kernel Analysis", Microscope)}
        </nav>

        {/* Context Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-slate-800">
          {isAnalysis && (
            <>
              <SectionTitle>Laboratory Experiments</SectionTitle>
              {Object.entries(analysisData).map(([id, op]) => {
                const isExpanded = expandedOp === id;
                const isOpActive = opId === id && !kernelId;

                return (
                  <div key={id} className="space-y-1">
                    <div
                      className={[
                        "w-full flex items-center justify-between rounded-xl transition-all font-bold text-sm",
                        isOpActive
                          ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20"
                          : "text-slate-400 hover:bg-slate-800",
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
                            isOpActive ? "text-emerald-100" : "text-slate-500"
                          }`}
                        >
                          {op.category}
                        </span>
                      </Link>

                      <button
                        onClick={() => setExpandedOp(isExpanded ? null : id)}
                        className="p-3 hover:bg-white/10 rounded-r-xl"
                        type="button"
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
                          const isKernelActive = kernelId === k.id;

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
                              <span className="flex items-center gap-2">
                                <Zap
                                  size={12}
                                  className={
                                    isKernelActive
                                      ? "text-yellow-400 fill-yellow-400"
                                      : "text-slate-600"
                                  }
                                />
                                {k.label}
                              </span>

                              <div className="flex items-center gap-2">
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

          {isOps && (
            <>
              <SectionTitle>Operator Library</SectionTitle>
              {Object.keys(allOpsData).map((id) => (
                <Link
                  key={id}
                  to={`/compute/ops?op=${id}`}
                  onClick={onClose}
                  className="w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm mb-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                >
                  <div className="min-w-0 flex flex-col items-start text-left">
                    <span className="truncate tracking-tight">{id}</span>
                    <span className="text-[9px] mt-0.5 uppercase tracking-tighter font-black text-slate-500">
                      {allOpsData[id]?.category || "Uncategorized"}
                    </span>
                  </div>
                  <ChevronRight size={14} className="opacity-20" />
                </Link>
              ))}
            </>
          )}

          {isTheory && (
            <>
              <SectionTitle>Mathematical Specs</SectionTitle>
              {Object.keys(theoryByOpId).map((id) => (
                <Link
                  key={id}
                  to={`/compute/theory?op=${id}`}
                  onClick={onClose}
                  className="w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm mb-1 text-slate-400 hover:bg-slate-800 hover:text-slate-200"
                >
                  <div className="min-w-0 flex flex-col items-start text-left">
                    <span className="truncate tracking-tight">{id}</span>
                    <span className="text-[9px] mt-0.5 uppercase tracking-tighter font-black text-slate-500">
                      {theoryByOpId[id]?.subtitle || "Spec"}
                    </span>
                  </div>
                  <ChevronRight size={14} className="opacity-20" />
                </Link>
              ))}
            </>
          )}

 

          {isPipeline && (
            <>
              <SectionTitle>Pipeline Context</SectionTitle>
              <div className="px-3 py-6 text-center opacity-60">
                <GitMerge size={24} className="mx-auto mb-3 text-emerald-400" />
                <p className="text-[10px] uppercase font-black tracking-widest text-slate-500">
                  Execution Planning
                </p>
                <p className="mt-3 text-xs text-slate-500 leading-relaxed">
                  semantic constraint와 hardware execution path를 연결하는
                  compute 내부 실행 계획 단계입니다.
                </p>
              </div>
            </>
          )}

          {!isComputeHome && !isAnalysis && !isOps && !isTheory && !isPipeline && (
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
              Hardware Verified
            </span>
          </div>
          <p className="text-slate-600 font-medium leading-tight italic">
            "Turning CUDA Kernels into <br /> Measurable Science."
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