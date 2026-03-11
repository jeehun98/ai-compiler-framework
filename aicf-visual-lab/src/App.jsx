import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import AICFOverviewPage from "./pages/overview/AICFOverviewPage.jsx";

import ComputeOptimizationPage from "./pages/overview/ComputeOptimizationPage.jsx";
import MemoryOptimizationPage from "./pages/overview/MemoryOptimizationPage.jsx";

import TheoryPage from "./pages/overview/compute/TheoryPage.jsx";
import PipelinePage from "./pages/overview/compute/PipelinePage.jsx";
import OpsPage from "./pages/overview/compute/OpsPage.jsx";
import KernelAnalysisPage from "./pages/overview/compute/KernelAnalysisPage.jsx";

import MemoryMethodsPage from "./pages/overview/memory/MemoryMethodsPage.jsx";
import MemoryMethodDetailPage from "./pages/overview/memory/MemoryMethodDetailPage.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AICFOverviewPage />} />

        {/* top overview */}
        <Route path="/compute" element={<ComputeOptimizationPage />} />
        <Route path="/memory" element={<MemoryOptimizationPage />} />

        {/* compute */}
        <Route path="/compute/theory" element={<TheoryPage />} />
        <Route path="/compute/pipeline" element={<PipelinePage />} />
        <Route path="/compute/ops" element={<OpsPage />} />
        <Route path="/compute/analysis" element={<KernelAnalysisPage />} />
        <Route path="/compute/analysis/:opId" element={<KernelAnalysisPage />} />
        <Route
          path="/compute/analysis/:opId/:kernelId"
          element={<KernelAnalysisPage />}
        />

        {/* memory */}
        <Route path="/memory/methods" element={<MemoryMethodsPage />} />
        <Route
          path="/memory/methods/:methodId"
          element={<MemoryMethodDetailPage />}
        />
        <Route
          path="/memory/methods/:methodId/:phaseId"
          element={<MemoryMethodDetailPage />}
        />

        {/* temporary: pipeline page not created yet */}
        <Route path="/memory/pipeline" element={<MemoryOptimizationPage />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}