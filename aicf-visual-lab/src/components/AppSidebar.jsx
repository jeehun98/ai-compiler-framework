// src/components/AppSidebar.jsx
import React, { useMemo } from "react";
import { Link, useLocation } from "react-router-dom";
import { Cpu, Terminal, LayoutDashboard, BookOpen, ArrowUpRight } from "lucide-react";

import { allOpsData } from "../data/index.js";

export default function AppSidebar({
  isOpen,
  onClose,
  activeOpId,
  version = "v1.0.4 Stable",
}) {
  const location = useLocation();
  const isOps = location.pathname === "/ops";

  const opIds = useMemo(() => Object.keys(allOpsData || {}), []);

  const navItem = (to, label, Icon) => {
    const isActive = location.pathname === to;
    return (
      <Link
        to={to}
        className={[
          "flex items-center gap-3 px-3 py-2.5 rounded-xl transition font-bold text-sm",
          isActive
            ? "bg-blue-600/10 text-blue-400 border border-blue-500/20"
            : "text-slate-400 hover:bg-slate-800 hover:text-white",
        ].join(" ")}
        onClick={onClose}
      >
        <Icon size={18} /> {label}
      </Link>
    );
  };

  return (
    <>
      {/* Overlay */}
      {isOpen && (
        <div
          className="fixed inset-0 z-40 bg-black/60 md:hidden backdrop-blur-sm"
          onClick={onClose}
          aria-hidden="true"
        />
      )}

      <aside
        className={[
          "fixed md:static inset-y-0 left-0 z-50 md:z-10",
          "w-[85vw] max-w-[320px] md:w-80",
          "bg-[#1e293b] border-r border-slate-800",
          "flex flex-col shadow-2xl transition-transform duration-300",
          isOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0",
        ].join(" ")}
        role="dialog"
        aria-modal={isOpen ? "true" : "false"}
      >
        {/* Logo */}
        <div className="p-6 border-b border-slate-800 bg-[#0f172a]/50">
          <Link to="/" className="flex items-center gap-3 group" onClick={onClose}>
            <div className="bg-blue-600 p-2 rounded-xl group-hover:bg-blue-500 transition">
              <Cpu size={20} className="text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg font-black tracking-tight text-white leading-none truncate">
                AICF LAB
              </h1>
              <span className="text-[10px] text-slate-500 font-bold uppercase tracking-widest">
                {version}
              </span>
            </div>
          </Link>
        </div>

        {/* Nav */}
        <nav className="p-4 space-y-1 border-b border-slate-800">
          <p className="px-3 text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">
            Navigation
          </p>

          {navItem("/", "Dashboard", LayoutDashboard)}
          {navItem("/ops", "Ops Explorer", Terminal)}
          {navItem("/theory", "Theory", BookOpen)}
        </nav>

        {/* Ops List (Ops 페이지에서만 노출) */}
        <div className="flex-1 overflow-y-auto p-4 space-y-1 scrollbar-thin scrollbar-thumb-slate-700">
          {isOps ? (
            <>
              <p className="px-3 text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2">
                Available Operators
              </p>

              {opIds.map((id) => {
                const active = activeOpId === id;
                return (
                  <Link
                    key={id}
                    to={`/ops?op=${id}`}
                    onClick={onClose}
                    className={[
                      "w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm",
                      active
                        ? "bg-blue-600 text-white shadow-lg"
                        : "text-slate-400 hover:bg-slate-800 hover:text-slate-200",
                    ].join(" ")}
                  >
                    <div className="min-w-0 flex flex-col items-start text-left">
                      <span className="truncate w-full">{id}</span>
                      <span
                        className={[
                          "text-[10px] mt-0.5 truncate w-full",
                          active ? "text-blue-100/90" : "text-slate-500",
                        ].join(" ")}
                      >
                        {allOpsData?.[id]?.category ?? "연산자 분류"}
                      </span>
                    </div>

                    {active ? (
                      <ArrowUpRight size={14} />
                    ) : (
                      <ArrowUpRight size={14} className="opacity-25" />
                    )}
                  </Link>
                );
              })}
            </>
          ) : (
            <div className="px-3 py-6 text-xs text-slate-500 leading-relaxed">
              <div className="text-[10px] font-black uppercase tracking-widest text-slate-600 mb-2">
                Tip
              </div>
              Ops Explorer에서 연산 리스트가 나타납니다.
            </div>
          )}
        </div>

        <div className="p-6 border-t border-slate-800 text-[10px] text-slate-600 font-medium">
          © 2026 AICF Compiler Team. <br /> Semantic Preserving Engine.
        </div>
      </aside>

      <style jsx="true">{`
        .scrollbar-thin::-webkit-scrollbar { width: 8px; }
        .scrollbar-thin::-webkit-scrollbar-thumb { background: #334155; border-radius: 999px; }
        .scrollbar-thin::-webkit-scrollbar-track { background: transparent; }
      `}</style>
    </>
  );
}