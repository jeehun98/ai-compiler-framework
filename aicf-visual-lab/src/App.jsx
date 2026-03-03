import React from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import HomePage from "./pages/HomePage.jsx";
import OpsPage from "./pages/OpsPage.jsx";
import TheoryPage from "./pages/TheoryPage.jsx"

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<HomePage />} />
        <Route path="/ops" element={<OpsPage />} />
        <Route path="/theory" element={<TheoryPage />} />
        {/* 나중 확장 */}
        {/* <Route path="/ir" element={<IRPage />} /> */}
        {/* <Route path="/kernels" element={<KernelPage />} /> */}

        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}