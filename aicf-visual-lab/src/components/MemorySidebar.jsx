import React, { useEffect, useState } from "react";
import { Link, useLocation, useParams } from "react-router-dom";
import {
  HardDrive,
  ChevronRight,
  ChevronDown,
  ShieldCheck,
  Activity,
  RotateCcw,
  Maximize2,
  LayoutDashboard,
  ArrowUpRight,
  GitMerge,
  Zap,
  Library,
} from "lucide-react";

export default function MemorySidebar({
  isOpen,
  onClose,
  version = "v1.0.6 Lab-Ready",
}) {
  const location = useLocation();
  const pathname = location.pathname;
  const { methodId, phaseId } = useParams();

  const methods = {
    "online-norm": {
      label: "Online Reducible Norm",
      category: "Single-Pass Reduction",
      icon: <Activity size={18} />,
      phases: ["theory", "hardware", "compiler"],
    },
    "weighted-reduction": {
      label: "Streaming Weighted Reduction",
      category: "Flash-Style Optimization",
      icon: <Zap size={18} />,
      phases: ["theory", "hardware", "compiler"],
    },
    rematerialization: {
      label: "Re-materializable Intermediate",
      category: "VRAM Saving Strategy",
      icon: <RotateCcw size={18} />,
      phases: ["theory", "hardware", "compiler"],
    },
    "tile-compatible": {
      label: "Tile-Compatible Compute",
      category: "SRAM Residency Planning",
      icon: <Maximize2 size={18} />,
      phases: ["theory", "hardware", "compiler"],
    },
  };

  const phaseLabels = {
    theory: "Math & Logic",
    hardware: "Physical Analysis",
    compiler: "MCIR Implementation",
  };

  const isOverview = pathname === "/memory";
  const isMethodsRoot = pathname === "/memory/methods";
  const isMethodsSection = pathname.startsWith("/memory/methods");
  const isPipeline = pathname.startsWith("/memory/pipeline");

  const [methodsOpen, setMethodsOpen] = useState(isMethodsSection);
  const [expandedMethod, setExpandedMethod] = useState(methodId || null);

  useEffect(() => {
    if (isMethodsSection) setMethodsOpen(true);
  }, [isMethodsSection]);

  useEffect(() => {
    if (methodId) {
      setMethodsOpen(true);
      setExpandedMethod(methodId);
    }
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

        {/* Navigation */}
        <nav className="p-4 space-y-1 border-b border-slate-800">
          <SectionTitle>Navigation</SectionTitle>

          {navItem("/memory", "Overview", LayoutDashboard, { exact: true })}

          <div className="space-y-1">
            <div
              className={[
                "w-full flex items-center justify-between rounded-xl transition-all font-bold text-sm",
                isMethodsRoot
                  ? "bg-emerald-600/10 text-emerald-400 border border-emerald-500/20"
                  : "text-slate-400 hover:bg-slate-800 hover:text-white",
              ].join(" ")}
            >
              <Link
                to="/memory/methods"
                onClick={onClose}
                className="flex-1 flex items-center gap-3 px-3 py-2.5"
              >
                <Library size={18} />
                Optimization Methods
              </Link>

              <button
                type="button"
                onClick={() => setMethodsOpen((v) => !v)}
                className="p-3 hover:bg-white/10 rounded-r-xl transition-colors"
              >
                {methodsOpen ? (
                  <ChevronDown size={14} />
                ) : (
                  <ChevronRight size={14} />
                )}
              </button>
            </div>

            {methodsOpen && (
              <div className="ml-5 pl-4 border-l border-slate-800 space-y-1 animate-in slide-in-from-top-2 duration-200">
                {Object.entries(methods).map(([id, method]) => {
                  const isExpanded = expandedMethod === id;
                  const isMethodRootActive = methodId === id && !phaseId;
                  const isMethodInPath = methodId === id;

                  return (
                    <div key={id} className="space-y-1">
                      <div
                        className={[
                          "w-full flex items-center justify-between rounded-xl transition-all text-sm font-bold",
                          isMethodRootActive
                            ? "bg-emerald-600 text-white shadow-lg shadow-emerald-600/20"
                            : isMethodInPath
                            ? "bg-slate-800/70 text-slate-200"
                            : "text-slate-500 hover:bg-slate-800 hover:text-slate-200",
                        ].join(" ")}
                      >
                        <Link
                          to={`/memory/methods/${id}`}
                          onClick={onClose}
                          className="flex-1 text-left flex flex-col px-4 py-3 min-w-0"
                        >
                          <span className="flex items-center gap-2 truncate">
                            {method.icon}
                            {method.label}
                          </span>
                          <span
                            className={`text-[9px] uppercase font-black mt-0.5 ${
                              isMethodRootActive
                                ? "text-emerald-100"
                                : isMethodInPath
                                ? "text-slate-400"
                                : "text-slate-600"
                            }`}
                          >
                            {method.category}
                          </span>
                        </Link>

                        <button
                          type="button"
                          onClick={() =>
                            setExpandedMethod(isExpanded ? null : id)
                          }
                          className="p-3 hover:bg-white/10 rounded-r-xl transition-colors"
                        >
                          {isExpanded ? (
                            <ChevronDown size={14} />
                          ) : (
                            <ChevronRight size={14} />
                          )}
                        </button>
                      </div>

                      {isExpanded && (
                        <div className="ml-5 pl-4 border-l border-slate-800 space-y-1 mt-1 animate-in slide-in-from-top-2 duration-200">
                          {method.phases.map((phase) => {
                            const isPhaseActive =
                              methodId === id && phaseId === phase;

                            return (
                              <Link
                                key={phase}
                                to={`/memory/methods/${id}/${phase}`}
                                onClick={onClose}
                                className={[
                                  "flex items-center justify-between px-3 py-2 rounded-lg text-[12px] font-bold transition-all",
                                  isPhaseActive
                                    ? "text-emerald-400 bg-emerald-400/5"
                                    : "text-slate-500 hover:text-slate-300 hover:bg-slate-800/50",
                                ].join(" ")}
                              >
                                <span>{phaseLabels[phase]}</span>
                                {isPhaseActive && (
                                  <ArrowUpRight
                                    size={12}
                                    className="animate-in fade-in zoom-in"
                                  />
                                )}
                              </Link>
                            );
                          })}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {navItem("/memory/pipeline", "Residency Pipeline", GitMerge)}
        </nav>

        {/* Footer */}
        <div className="mt-auto p-6 border-t border-slate-800 bg-[#0b0f1a] text-[10px]">
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