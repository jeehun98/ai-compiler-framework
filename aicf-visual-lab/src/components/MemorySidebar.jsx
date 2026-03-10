import React from "react";
import { Link, useLocation } from "react-router-dom";
import {
  LayoutDashboard,
  HardDrive,
  ChevronRight,
  ShieldCheck,
  Layers,
  Waypoints,
  GitBranch,
  Database,
} from "lucide-react";

export default function MemorySidebar({
  isOpen,
  onClose,
  version = "v1.0.6 Lab-Ready",
}) {
  const location = useLocation();
  const pathname = location.pathname;

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
        <div className="p-6 border-b border-slate-800 bg-[#0b0f1a]">
          <Link
            to="/"
            className="flex items-center gap-3 group"
            onClick={onClose}
          >
            <div className="bg-emerald-600 p-2 rounded-xl group-hover:bg-emerald-500 transition shadow-lg shadow-emerald-600/20">
              <HardDrive size={20} className="text-white" />
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

        <nav className="p-4 space-y-1 border-b border-slate-800">
          <SectionTitle>Navigation</SectionTitle>
          {navItem("/memory", "Overview", LayoutDashboard, { exact: true })}
        </nav>

        <div className="flex-1 overflow-y-auto p-4 space-y-2 scrollbar-thin scrollbar-thumb-slate-800">
          <SectionTitle>Memory Sections</SectionTitle>

          <div className="w-full flex items-center justify-between px-4 py-3 rounded-xl border border-emerald-500/10 bg-emerald-500/5 text-emerald-300">
            <div className="flex flex-col items-start text-left">
              <span className="tracking-tight font-bold">
                Memory Optimization
              </span>
              <span className="text-[9px] mt-0.5 uppercase tracking-tighter font-black text-emerald-500/70">
                Current Axis
              </span>
            </div>
            <HardDrive size={16} />
          </div>

          {[
            {
              icon: <Waypoints size={16} />,
              title: "Residency Model",
              desc: "값을 가능한 오래 온칩에 머물게 하는 실행 관점",
            },
            {
              icon: <GitBranch size={16} />,
              title: "Boundary Elimination",
              desc: "operator 경계를 물리 실행 경계와 분리하는 사고",
            },
            {
              icon: <Database size={16} />,
              title: "Traffic-Aware Planning",
              desc: "HBM write/read를 최소화하는 dataflow 계획",
            },
          ].map((item) => (
            <div
              key={item.title}
              className="w-full rounded-xl px-4 py-4 text-slate-400 border border-slate-800 bg-[#111827]"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2 text-emerald-400 font-bold text-sm">
                  {item.icon}
                  {item.title}
                </div>
                <ChevronRight size={14} className="opacity-20" />
              </div>
              <p className="mt-3 text-xs leading-relaxed text-slate-500">
                {item.desc}
              </p>
            </div>
          ))}

          <SectionTitle>Axis Summary</SectionTitle>
          <div className="px-3 py-6 text-center opacity-80">
            <Layers size={24} className="mx-auto mb-3 text-emerald-400" />
            <p className="text-[10px] uppercase font-black tracking-widest text-slate-500">
              Same Result, Less Movement
            </p>
            <p className="mt-3 text-xs text-slate-500 leading-relaxed">
              Memory Optimization은 더 많은 연산이 아니라 더 적은 이동을 목표로
              합니다. residency, fusion, traffic control이 이 축의 핵심입니다.
            </p>
          </div>
        </div>

        <div className="p-6 border-t border-slate-800 bg-[#0b0f1a] text-[10px]">
          <div className="flex items-center gap-2 mb-2 text-emerald-500">
            <ShieldCheck size={12} strokeWidth={3} />
            <span className="font-black uppercase tracking-widest">
              Traffic Aware
            </span>
          </div>
          <p className="text-slate-600 font-medium leading-tight italic">
            "Optimization begins when <br /> movement becomes visible."
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