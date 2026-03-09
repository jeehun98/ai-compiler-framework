import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import AICFOverviewPage from "./pages/AICFOverviewPage.jsx";
import ComputeOptimizationPage from "./pages/ComputeOptimizationPage.jsx";
import MemoryOptimizationPage from "./pages/MemoryOptimizationPage.jsx";
import TheoryPage from "./pages/TheoryPage.jsx";
import PipelinePage from "./pages/PipelinePage.jsx";
import OpsPage from "./pages/OpsPage.jsx";
import KernelAnalysisPage from "./pages/KernelAnalysisPage.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* top level */}
        <Route path="/" element={<AICFOverviewPage />} />
        <Route path="/compute" element={<ComputeOptimizationPage />} />
        <Route path="/memory" element={<MemoryOptimizationPage />} />

        {/* compute sub pages */}
        <Route path="/compute/theory" element={<TheoryPage />} />
        <Route path="/compute/pipeline" element={<PipelinePage />} />
        <Route path="/compute/ops" element={<OpsPage />} />

        <Route path="/compute/analysis" element={<KernelAnalysisPage />} />
        <Route path="/compute/analysis/:opId" element={<KernelAnalysisPage />} />
        <Route
          path="/compute/analysis/:opId/:kernelId"
          element={<KernelAnalysisPage />}
        />

        {/* legacy redirects */}
        <Route path="/overview" element={<Navigate to="/" replace />} />
        <Route path="/theory" element={<Navigate to="/compute/theory" replace />} />
        <Route path="/pipeline" element={<Navigate to="/compute/pipeline" replace />} />
        <Route path="/ops" element={<Navigate to="/compute/ops" replace />} />
        <Route path="/analysis" element={<Navigate to="/compute/analysis" replace />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}