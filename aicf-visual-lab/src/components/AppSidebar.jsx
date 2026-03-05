import React, { useMemo } from "react";
import { Link, useLocation } from "react-router-dom";
import {
  Cpu,
  Terminal,
  LayoutDashboard,
  BookOpen,
  ArrowUpRight,
  ChevronRight,
  GitMerge,
  Settings2,
  ShieldCheck,
  Microscope // 분석용 아이콘 추가
} from "lucide-react";

import { allOpsData } from "../data/index.js";
import { theoryByOpId } from "../data/theory/index.js";

export default function AppSidebar({
  isOpen,
  onClose,
  activeOpId,
  version = "v1.0.4 Stable",
}) {
  const location = useLocation();
  const isOps = location.pathname === "/ops";
  const isTheory = location.pathname === "/theory";
  const isPipeline = location.pathname === "/pipeline";
  const isAnalysis = location.pathname.startsWith("/analysis"); // Analysis 페이지 판별

  // ✅ 수정: Analysis 페이지에서도 커널 리스트(allOpsData 기반)를 보여주도록 설정
  const listIds = useMemo(() => {
    if (isTheory) return Object.keys(theoryByOpId || {});
    if (isOps || isAnalysis) return Object.keys(allOpsData || {}); // 분석 페이지도 리스트 공유
    return [];
  }, [isOps, isTheory, isAnalysis]);

  const navItem = (to, label, Icon) => {
    // pathname이 정확히 일치하거나, 해당 경로로 시작하는 경우(하위 경로 포함) active 처리
    const isActive = to === "/" ? location.pathname === "/" : location.pathname.startsWith(to);
    
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

  const SectionTitle = ({ children }) => (
    <p className="px-3 text-[10px] font-black text-slate-500 uppercase tracking-widest mb-2 mt-4 first:mt-0">
      {children}
    </p>
  );

  return (
    <>
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
      >
        <div className="p-6 border-b border-slate-800 bg-[#0f172a]/50">
          <Link to="/" className="flex items-center gap-3 group" onClick={onClose}>
            <div className="bg-blue-600 p-2 rounded-xl group-hover:bg-blue-500 transition shadow-lg shadow-blue-600/20">
              <Cpu size={20} className="text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="text-lg font-black tracking-tight text-white leading-none truncate uppercase">
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
          {navItem("/", "Dashboard", LayoutDashboard)}
          {navItem("/ops", "Ops Explorer", Terminal)}
          {navItem("/kernels", "Kernel Analysis", Microscope)} {/* 신규 추가 */}
          {navItem("/theory", "Theory Specs", BookOpen)}
          {navItem("/pipeline", "Compiler Pipeline", GitMerge)}
        </nav>

        <div className="flex-1 overflow-y-auto p-4 space-y-1 scrollbar-thin scrollbar-thumb-slate-700">
          {listIds.length > 0 ? (
            <>
              <SectionTitle>
                {isTheory ? "Mathematical Specs" : "Target Operators"}
              </SectionTitle>

              {listIds.map((id) => {
                const active = activeOpId === id;
                // 이동 경로 설정
                let to = isTheory ? `/theory?op=${id}` : isAnalysis ? `/analysis/${id}` : `/ops?op=${id}`;

                const category = isTheory
                  ? (theoryByOpId?.[id]?.subtitle ?? "Theoretical Object")
                  : (allOpsData?.[id]?.category ?? "Graph Node");

                return (
                  <Link
                    key={id}
                    to={to}
                    onClick={onClose}
                    className={[
                      "w-full flex items-center justify-between px-4 py-3 rounded-xl transition-all font-bold text-sm mb-1",
                      active
                        ? "bg-blue-600 text-white shadow-lg shadow-blue-600/20"
                        : "text-slate-400 hover:bg-slate-800 hover:text-slate-200",
                    ].join(" ")}
                  >
                    <div className="min-w-0 flex flex-col items-start text-left">
                      <span className="truncate w-full tracking-tight">{id}</span>
                      <span
                        className={[
                          "text-[9px] mt-0.5 truncate w-full uppercase tracking-tighter font-black",
                          active ? "text-blue-100/70" : "text-slate-500",
                        ].join(" ")}
                      >
                        {category}
                      </span>
                    </div>
                    {active ? <ArrowUpRight size={14} className="shrink-0" /> : <ChevronRight size={14} className="opacity-20 shrink-0" />}
                  </Link>
                );
              })}
            </>
          ) : (
            <div className="px-3 py-10 text-center animate-in fade-in duration-500">
              <div className="inline-flex p-3 bg-slate-800/50 rounded-2xl mb-4 text-slate-600">
                <Settings2 size={24} />
              </div>
              <p className="text-[11px] text-slate-500 font-medium leading-relaxed">
                {isPipeline 
                  ? "파이프라인 페이지는 전체 실행 <br/> 흐름 가이드를 제공합니다." 
                  : "상세 리스트가 필요한 <br/> 메뉴를 선택해주세요."}
              </p>
            </div>
          )}
        </div>

        <div className="p-6 border-t border-slate-800">
          <div className="flex items-center gap-2 mb-2 text-blue-500">
             <ShieldCheck size={12} strokeWidth={3} />
             <span className="text-[10px] font-black uppercase tracking-widest">Semantic Verified</span>
          </div>
          <div className="text-[10px] text-slate-600 font-medium leading-tight">
            © 2026 AICF Compiler Team. <br /> 
            High-Performance ML Engine.
          </div>
        </div>
      </aside>

      <style jsx="true">{`
        .scrollbar-thin::-webkit-scrollbar { width: 6px; }
        .scrollbar-thin::-webkit-scrollbar-thumb { background: #334155; border-radius: 999px; }
        .scrollbar-thin::-webkit-scrollbar-track { background: transparent; }
      `}</style>
    </>
  );
}