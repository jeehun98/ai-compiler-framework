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
  btnExport: document.getElementById("btnExport"),
  btnImport: document.getElementById("btnImport"),
  btnReset: document.getElementById("btnReset"),
  fileImport: document.getElementById("fileImport"),
};

let lab = loadLab();

function rerender(){
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

function boot(){
  // 첫 실행: bias_add 템플릿 하나 심어두면 시작이 편함
  if(Object.keys(lab.kernels).length === 0){
    seedBiasAdd(lab);
    saveLab(lab);
  }

  wireHeaderButtons({
    lab,
    btnNewKernel: els.btnNewKernel,
    btnReset: els.btnReset,
    btnExport: els.btnExport,
    btnImport: els.btnImport,
    fileImport: els.fileImport,
    onChange: rerender,
  });

  els.q.oninput = () => rerender();
  rerender();
}

boot();
