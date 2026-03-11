import React, { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import {
  HardDrive,
  ChevronRight,
  ChevronDown,
  ShieldCheck,
  LayoutDashboard,
  ArrowUpRight,
  GitMerge,
  Library,
  Zap,
  Database,
  Layers,
  Activity,
} from "lucide-react";
import {
  memoryMethods,
  memoryPhaseLabels,
} from "../data/memory/methods";

export default function MemorySidebar({
  isOpen,
  onClose,
  version = "v1.0.6 Lab-Ready",
}) {
  const location = useLocation();
  const { methodId, phaseId } = useParams();
  const pathname = location.pathname;

  const isMemoryHome = pathname === "/memory";
  const isMethods = pathname.startsWith("/memory/methods");
  const isPipeline = pathname.startsWith("/memory/pipeline");

  const [expandedMethod, setExpandedMethod] = useState(methodId || "online-norm");

  useEffect(() => {
    if (methodId) setExpandedMethod(methodId);
  }, [methodId]);

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
            <div className="bg-emerald-600 p-2 rounded-xl group-hover:bg-emerald-500 transition shadow-lg shadow-emerald-600/20 text-white">
              <HardDrive size={20} />
            </div>

            <div>
              <h1 className="text-lg font-black tracking-tight text-white leading-none">
                AICF MEMORY
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
          {navItem("/memory", "Overview", LayoutDashboard, { exact: true })}
          {navItem("/memory/methods", "Optimization Methods", Library)}
          {navItem("/memory/pipeline", "Residency Pipeline", GitMerge)}
        </nav>

        {/* Context Area */}
        <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-slate-800">
          {isMethods && (
            <>
              <SectionTitle>Method Library</SectionTitle>

              {memoryMethods.map((method) => {
                const NavIcon = method.navIcon;
                const isExpanded = expandedMethod === method.id;
                const isMethodActive = methodId === method.id && !phaseId;
                const isMethodPath = methodId === method.id;

                return (
                  <div key={method.id} className="space-y-1">
                    <div
                      className={[
                        "w-full flex items-center justify-between rounded-xl transition-all font-bold text-sm",
                        isMethodActive
                          ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20"
                          : isMethodPath
                          ? "bg-slate-800/70 text-slate-200"
                          : "text-slate-400 hover:bg-slate-800",
                      ].join(" ")}
                    >
                      <Link
                        to={`/memory/methods/${method.id}`}
                        className="flex-1 text-left flex flex-col px-4 py-3"
                        onClick={onClose}
                      >
                        <span className="flex items-center gap-2">
                          <NavIcon size={16} />
                          {method.label}
                          {isMethodActive && (
                            <Zap
                              size={10}
                              className="text-yellow-300 animate-pulse fill-yellow-300"
                            />
                          )}
                        </span>

                        <span
                          className={`text-[9px] uppercase font-black ${
                            isMethodActive
                              ? "text-emerald-100"
                              : isMethodPath
                              ? "text-slate-400"
                              : "text-slate-500"
                          }`}
                        >
                          {method.category}
                        </span>
                      </Link>

                      <button
                        type="button"
                        onClick={() =>
                          setExpandedMethod(isExpanded ? null : method.id)
                        }
                        className="p-3 hover:bg-white/10 rounded-r-xl"
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
                        {method.phases.map((phase) => {
                          const isPhaseActive =
                            methodId === method.id && phaseId === phase;

                          return (
                            <Link
                              key={phase}
                              to={`/memory/methods/${method.id}/${phase}`}
                              onClick={onClose}
                              className={[
                                "flex items-center justify-between px-3 py-2 rounded-lg text-[13px] font-bold transition-all",
                                isPhaseActive
                                  ? "text-emerald-400 bg-emerald-400/5"
                                  : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50",
                              ].join(" ")}
                            >
                              <span className="flex items-center gap-2">
                                <Database
                                  size={12}
                                  className={
                                    isPhaseActive
                                      ? "text-emerald-400"
                                      : "text-slate-600"
                                  }
                                />
                                {memoryPhaseLabels[phase]}
                              </span>

                              <div className="flex items-center gap-2">
                                <span className="text-[8px] border border-slate-700 px-1.5 py-0.5 rounded uppercase tracking-tighter opacity-60">
                                  {phase}
                                </span>
                                {isPhaseActive && <ArrowUpRight size={12} />}
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
              <SectionTitle>Pipeline Context</SectionTitle>
              <div className="px-3 py-6 text-center opacity-60">
                <GitMerge size={24} className="mx-auto mb-3 text-emerald-400" />
                <p className="text-[10px] uppercase font-black tracking-widest text-slate-500">
                  Residency Planning
                </p>
                <p className="mt-3 text-xs text-slate-500 leading-relaxed">
                  global memory round-trip을 줄이기 위해
                  streaming execution, re-materialization, tiling residency를
                  설계하는 memory 내부 계획 단계입니다.
                </p>
              </div>
            </>
          )}

          {isMemoryHome && (
            <>
              <SectionTitle>Memory Context</SectionTitle>
              <div className="space-y-3">
                <div className="w-full flex items-center justify-between px-4 py-3 rounded-xl border border-emerald-500/10 bg-emerald-500/5 text-emerald-300">
                  <div className="flex flex-col items-start text-left">
                    <span className="tracking-tight font-bold">
                      Residency Engine
                    </span>
                    <span className="text-[9px] mt-0.5 uppercase tracking-tighter font-black text-emerald-500/70">
                      Active Traffic Control
                    </span>
                  </div>
                  <Activity size={16} className="animate-pulse" />
                </div>

                <div className="px-3 py-6 text-center opacity-60">
                  <Layers size={24} className="mx-auto mb-3 text-emerald-400" />
                  <p className="text-[10px] uppercase font-black tracking-widest text-slate-500">
                    Memory Optimization Domain
                  </p>
                  <p className="mt-3 text-xs text-slate-500 leading-relaxed">
                    계산 자체보다 데이터 이동을 먼저 다룹니다.
                    어떤 값이 저장되어야 하는지, 어떤 값이 다시 계산될 수
                    있는지, 어떤 reduction이 streaming 가능한지를 정의합니다.
                  </p>
                </div>
              </div>
            </>
          )}

          {!isMemoryHome && !isMethods && !isPipeline && (
            <div className="px-3 py-10 text-center opacity-40">
              <Layers size={24} className="mx-auto mb-2" />
              <p className="text-[10px] uppercase font-black tracking-widest text-slate-500">
                No Active Memory Context
              </p>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-6 border-t border-slate-800 bg-[#0b0f1a] text-[10px]">
          <div className="flex items-center gap-2 mb-2 text-emerald-500">
            <ShieldCheck size={12} strokeWidth={3} />
            <span className="font-black uppercase tracking-widest">
              Traffic Aware
            </span>
          </div>

          <p className="text-slate-600 font-medium leading-tight italic">
            "Eliminating HBM round-trips <br /> through physical residency."
          </p>
        </div>

        <style jsx="true">{`
          .scrollbar-thin::-webkit-scrollbar {
            width: 4px;
          }
          .scrollbar-thin::-webkit-scrollbar-thumb {
            background: #1f2937;
            border-radius: 10px;
          }
        `}</style>
      </aside>
    </>
  );
}