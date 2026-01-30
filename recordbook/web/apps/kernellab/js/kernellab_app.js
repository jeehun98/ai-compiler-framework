import { loadLab, saveLab, loadKernelJson, upsertKernel } from "./kernellab_state.js";
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
  fileImportKernel: document.getElementById("fileImportKernel")
};

let lab = loadLab();

function rerender() {
  renderCatalog({
    catalogEl: els.catalog,
    qEl: els.q,
    kmetaEl: els.kmeta,
    lab,
    onChange: rerender
  });

  renderBlocks({
    blocksEl: els.blocks,
    inspectorEl: els.inspector,
    quickEl: els.quick,
    lab,
    onChange: rerender
  });

  saveLab(lab);
}

// 최초 boot에서 라이브러리 로드
async function ensureLibrarySeeded() {
  if (Object.keys(lab.kernels).length > 0) return;

  // apps/kernellab/index.html 기준
  const indexPath = "./data/kernels/index.json";
  let files = null;

  try {
    files = await loadKernelJson(indexPath);
  } catch (e) {
    // index.json이 없으면 아무것도 안 넣음 (수동 import로 시작 가능)
    console.warn("Kernel library index missing:", e);
    return;
  }

  if (!Array.isArray(files)) return;

  for (const fn of files) {
    try {
      const k = await loadKernelJson(`./data/kernels/${fn}`);
      upsertKernel(lab, k);
    } catch (e) {
      console.warn("Failed to load kernel:", fn, e);
    }
  }

  // selection default
  const first = Object.keys(lab.kernels)[0] || null;
  lab.selectedKernelId = first;
  lab.selectedBlockId = "overview";
  saveLab(lab);
}

async function boot() {
  await ensureLibrarySeeded();

  wireHeaderButtons({
    lab,
    btnNewKernel: els.btnNewKernel,
    btnExportKernel: els.btnExportKernel,
    btnImportKernel: els.btnImportKernel,
    btnReset: els.btnReset,
    fileImportKernel: els.fileImportKernel,
    onChange: rerender
  });

  els.q.oninput = () => rerender();
  rerender();
}

boot();
