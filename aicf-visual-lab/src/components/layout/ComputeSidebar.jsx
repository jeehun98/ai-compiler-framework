import React from "react";
import { Link, useLocation } from "react-router-dom";
import { allOpsData } from "../../data/ops/index.js";
import { theoryPropertyGroups } from "../../data/theory/properties/index.js";

import {
  Terminal,
  BookOpen,
  ChevronRight,
  Zap,
  Beaker,
  Layers,
  ShieldCheck,
  Boxes,
  RefreshCw,
  GitBranch,
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
    <p className="mt-4 mb-2 px-3 text-[10px] font-black uppercase tracking-widest text-slate-500 first:mt-0">
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
          "flex items-center gap-3 rounded-xl border px-3 py-2.5 text-sm font-bold transition",
          isActive
            ? "border-emerald-500/20 bg-emerald-600/10 text-emerald-400"
            : "border-transparent text-slate-400 hover:bg-slate-800 hover:text-white",
        ].join(" ")}
      >
        <Icon size={18} />
        {label}
      </Link>
    );
  };

  const groupMeta = {
    foundational: {
      title: "Foundational",
      icon: Boxes,
      sectionClass: "text-blue-400",
      chipClass:
        "border-blue-500/20 bg-blue-500/5 text-blue-300",
      itemActiveClass:
        "border-blue-500/20 bg-blue-600/10 text-blue-300",
      itemActiveSubClass: "text-blue-200/80",
      chevronClass: "text-blue-400",
    },
    reconstructive: {
      title: "Reconstructive",
      icon: RefreshCw,
      sectionClass: "text-purple-400",
      chipClass:
        "border-purple-500/20 bg-purple-500/5 text-purple-300",
      itemActiveClass:
        "border-purple-500/20 bg-purple-600/10 text-purple-300",
      itemActiveSubClass: "text-purple-200/80",
      chevronClass: "text-purple-400",
    },
    structural: {
      title: "Structural",
      icon: GitBranch,
      sectionClass: "text-amber-400",
      chipClass:
        "border-amber-500/20 bg-amber-500/5 text-amber-300",
      itemActiveClass:
        "border-amber-500/20 bg-amber-600/10 text-amber-300",
      itemActiveSubClass: "text-amber-200/80",
      chevronClass: "text-amber-400",
    },
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
          "fixed inset-y-0 left-0 z-50 flex w-[85vw] max-w-[320px] flex-col border-r border-slate-800 bg-[#0f172a] shadow-2xl transition-transform duration-300 md:static md:z-10 md:w-80",
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        ].join(" ")}
      >
        <div className="border-b border-slate-800 bg-[#0b0f1a] p-6">
          <Link
            to="/"
            className="group flex items-center gap-3"
            onClick={onClose}
          >
            <div className="rounded-xl bg-emerald-600 p-2 shadow-lg shadow-emerald-600/20 transition group-hover:bg-emerald-500">
              <Beaker size={20} className="text-white" />
            </div>
            <div>
              <h1 className="text-lg font-black leading-none tracking-tight text-white">
                AICF COMPUTE
              </h1>
              <span className="text-[10px] font-bold uppercase tracking-widest text-slate-500">
                {version}
              </span>
            </div>
          </Link>
        </div>

        <nav className="space-y-1 border-b border-slate-800 p-4">
          <SectionTitle>Navigation</SectionTitle>
          {navItem("/compute", "Overview", Layers, { exact: true })}
          {navItem("/compute/theory", "Property Atlas", BookOpen)}
          {navItem("/compute/ops", "Ops Explorer", Terminal)}
        </nav>

        <div className="scrollbar-thin flex-1 overflow-y-auto space-y-2 p-4 scrollbar-thumb-slate-800">
          {isTheory && (
            <>
              <SectionTitle>Property Atlas</SectionTitle>

              {theoryPropertyGroups.map((group) => {
                const meta = groupMeta[group.id] ?? groupMeta.foundational;
                const GroupIcon = meta.icon;

                return (
                  <div key={group.id} className="mb-4">
                    <div className="mb-2 flex items-center gap-2 px-3">
                      <div
                        className={[
                          "inline-flex items-center gap-1.5 rounded-xl border px-2.5 py-1 text-[9px] font-black uppercase tracking-widest",
                          meta.chipClass,
                        ].join(" ")}
                      >
                        <GroupIcon size={10} />
                        {meta.title}
                      </div>
                    </div>

                    <div className="space-y-1">
                      {group.items.map((spec) => {
                        const isSelected = selectedTheoryPropertyId === spec.id;

                        return (
                          <Link
                            key={spec.id}
                            to={`/compute/theory?property=${spec.id}`}
                            onClick={onClose}
                            className={[
                              "mb-1 flex w-full items-center justify-between rounded-xl border px-4 py-3 text-sm font-bold transition-all",
                              isSelected
                                ? meta.itemActiveClass
                                : "border-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-200",
                            ].join(" ")}
                          >
                            <div className="min-w-0 flex flex-col items-start text-left">
                              <span className="flex items-center gap-2 truncate tracking-tight">
                                {spec.id}
                                {isSelected && (
                                  <Zap
                                    size={10}
                                    className="animate-pulse fill-yellow-300 text-yellow-300"
                                  />
                                )}
                              </span>

                              <span
                                className={[
                                  "mt-0.5 text-[9px] font-black uppercase tracking-tighter",
                                  isSelected
                                    ? meta.itemActiveSubClass
                                    : "text-slate-500",
                                ].join(" ")}
                              >
                                {spec?.subtitle || "Property"}
                              </span>
                            </div>

                            <ChevronRight
                              size={14}
                              className={
                                isSelected ? meta.chevronClass : "opacity-20"
                              }
                            />
                          </Link>
                        );
                      })}
                    </div>
                  </div>
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
                      "mb-1 flex w-full items-center justify-between rounded-xl border px-4 py-3 text-sm font-bold transition-all",
                      isSelected
                        ? "border-emerald-500/20 bg-emerald-600/10 text-emerald-400"
                        : "border-transparent text-slate-400 hover:bg-slate-800 hover:text-slate-200",
                    ].join(" ")}
                  >
                    <div className="min-w-0 flex flex-col items-start text-left">
                      <span className="flex items-center gap-2 truncate tracking-tight">
                        {id}
                        {isSelected && (
                          <Zap
                            size={10}
                            className="animate-pulse fill-yellow-300 text-yellow-300"
                          />
                        )}
                      </span>

                      <span
                        className={`mt-0.5 text-[9px] font-black uppercase tracking-tighter ${
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

              <div className="space-y-4 rounded-2xl border border-slate-800 bg-[#111827] p-4">
                <div>
                  <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">
                    Structure
                  </p>

                  <div className="mt-3 space-y-2">
                    {[
                      "Theory defines semantic properties",
                      "Properties are grouped as foundational, reconstructive, and structural",
                      "Ops maps operators to property profiles",
                      "Compute focuses on invariant-preserving execution",
                    ].map((line) => (
                      <div
                        key={line}
                        className="flex items-start gap-2 text-sm text-slate-400"
                      >
                        <div className="mt-1.5 h-1.5 w-1.5 rounded-full bg-emerald-500" />
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
              <p className="text-[10px] font-black uppercase tracking-widest text-slate-500">
                No Active Compute Context
              </p>
            </div>
          )}
        </div>

        <div className="border-t border-slate-800 bg-[#0b0f1a] p-6 text-[10px]">
          <div className="mb-2 flex items-center gap-2 text-emerald-500">
            <ShieldCheck size={12} strokeWidth={3} />
            <span className="font-black uppercase tracking-widest">
              Semantic Boundary
            </span>
          </div>

          <p className="font-medium italic leading-tight text-slate-600">
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