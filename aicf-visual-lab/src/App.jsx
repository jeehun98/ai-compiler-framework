import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import AICFOverviewPage from "./pages/overview/AICFOverviewPage.jsx";

import ComputeOverviewPage from "./pages/overview/ComputeOverviewPage.jsx";
import MemoryOverviewPage from "./pages/overview/MemoryOverviewPage.jsx";
import AICFLabPage from "./pages/overview/AICFLabPage.jsx";

import PropertyPage from "./pages/overview/compute/PropertyPage.jsx";
import OpsPage from "./pages/overview/compute/OpsPage.jsx";

import MemoryMethodsPage from "./pages/overview/memory/MemoryMethodsPage.jsx";
import MemoryMethodDetailPage from "./pages/overview/memory/MemoryMethodDetailPage.jsx";

import LabAnalysisPage from "./pages/overview/lab/LabAnalysisPage.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AICFOverviewPage />} />

        {/* top overview */}
        <Route path="/compute" element={<ComputeOverviewPage />} />
        <Route path="/memory" element={<MemoryOverviewPage />} />
        <Route path="/lab" element={<AICFLabPage />} />

        {/* compute */}
        <Route path="/compute/property" element={<PropertyPage />} />
        <Route path="/compute/ops" element={<OpsPage />} />

        {/* memory */}
        <Route path="/memory/methods" element={<MemoryMethodsPage />} />
        <Route
          path="/memory/methods/:methodId"
          element={<MemoryMethodDetailPage />}
        />

        {/* lab */}
        <Route path="/lab/analysis" element={<LabAnalysisPage />} />
        <Route path="/lab/analysis/:opId" element={<LabAnalysisPage />} />
        <Route
          path="/lab/analysis/:opId/:kernelId"
          element={<LabAnalysisPage />}
        />

        {/* temporary alias until pipeline page is implemented */}
        <Route path="/memory/pipeline" element={<MemoryOverviewPage />} />

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}