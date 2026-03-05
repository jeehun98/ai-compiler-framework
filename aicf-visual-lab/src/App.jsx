import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import HomePage from "./pages/HomePage.jsx";
import OpsPage from "./pages/OpsPage.jsx";
import TheoryPage from "./pages/TheoryPage.jsx"
import PipelinePage from "./pages/PipelinePage.jsx";
import KernelAnalysisPage from "./pages/KernelAnalysisPage.jsx";

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/ops" element={<OpsPage />} />
        <Route path="/theory" element={<TheoryPage />} />
        <Route path="/pipeline" element={<PipelinePage />} />
        <Route path="/kernels" element={<KernelAnalysisPage/>}/>

        {/* 나중 확장 */}
        {/* <Route path="/ir" element={<IRPage />} /> */}
        {/* <Route path="/kernels" element={<KernelPage />} /> */}

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}