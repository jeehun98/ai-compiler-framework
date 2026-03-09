import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import HomePage from "./pages/HomePage.jsx";
import OpsPage from "./pages/OpsPage.jsx";
import TheoryPage from "./pages/TheoryPage.jsx"
import PipelinePage from "./pages/PipelinePage.jsx";
import KernelAnalysisPage from "./pages/KernelAnalysisPage.jsx";
import AICFOverviewPage from "./pages/AICFOverviewPage.jsx";


// src/App.jsx 수정본
export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/ops" element={<OpsPage />} />
        <Route path="/theory" element={<TheoryPage />} />
        <Route path="/pipeline" element={<PipelinePage />} />

        <Route path="/overview" element={<AICFOverviewPage />} />

        {/* ✅ 수정: 파라미터를 계층적으로 받을 수 있도록 설정 */}
        {/* 1. 기본 분석 대시보드 */}
        <Route path="/analysis" element={<KernelAnalysisPage />} />
        {/* 2. 연산별 비교 페이지 (예: /analysis/add) */}
        <Route path="/analysis/:opId" element={<KernelAnalysisPage />} />
        {/* 3. 개별 커널 상세 페이지 (예: /analysis/add/f16x2) */}
        <Route path="/analysis/:opId/:kernelId" element={<KernelAnalysisPage />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}