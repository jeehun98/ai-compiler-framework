import { loadLab, saveLab, seedBiasAdd } from "./kernellab_state.js";
import { renderCatalog, renderBlocks, wireHeaderButtons } from "./kernellab_ui.js";

const els = {
  q: document.getElementById("q"),
  catalog: document.getElementById("catalog"),
  blocks: document.getElementById("blocks"),
  inspector: document.getElementById("inspector"),
  quick: document.getElementById("quick"),
  kmeta: document.getElementById("kmeta"),

  btnNewKernel: document.getElementById("btnNewKernel"),
  btnExportKernel: document.getElementById("btnExportKernel"),
  btnImportKernel: document.getElementById("btnImportKernel"),
  btnReset: document.getElementById("btnReset"),
  fileImportKernel: document.getElementById("fileImportKernel"),
};

let lab = loadLab();

function rerender() {
  renderCatalog({
    catalogEl: els.catalog,
    qEl: els.q,
    kmetaEl: els.kmeta,
    lab,
    onChange: rerender,
  });

  renderBlocks({
    blocksEl: els.blocks,
    inspectorEl: els.inspector,
    quickEl: els.quick,
    lab,
    onChange: rerender,
  });

  saveLab(lab);
}

function boot() {
  // 첫 실행 시 기본 커널 하나
  if (Object.keys(lab.kernels).length === 0) {
    seedBiasAdd(lab);
    saveLab(lab);
  }

  wireHeaderButtons({
    lab,
    btnNewKernel: els.btnNewKernel,
    btnExportKernel: els.btnExportKernel,
    btnImportKernel: els.btnImportKernel,
    btnReset: els.btnReset,
    fileImportKernel: els.fileImportKernel,
    onChange: rerender,
  });

  els.q.oninput = () => rerender();
  rerender();
}

boot();
