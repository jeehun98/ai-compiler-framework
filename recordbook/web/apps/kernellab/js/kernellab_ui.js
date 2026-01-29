import { saveLab, newKernel, seedBiasAdd } from "./kernellab_state.js";
import { uid } from "../../graph/js/utils.js";

/* ---------- helpers ---------- */
function el(tag, cls, text) {
  const d = document.createElement(tag);
  if (cls) d.className = cls;
  if (text != null) d.textContent = text;
  return d;
}

function inputRow(label, value, oninput) {
  const row = el("div", "insp-row");
  row.appendChild(el("div", "insp-label", label));
  const inp = document.createElement("input");
  inp.className = "insp-input";
  inp.value = value || "";
  inp.oninput = (e) => oninput(e.target.value);
  row.appendChild(inp);
  return row;
}

function textareaRow(label, value, oninput) {
  const row = el("div", "insp-row");
  row.appendChild(el("div", "insp-label", label));
  const ta = document.createElement("textarea");
  ta.className = "insp-ta";
  ta.value = value || "";
  ta.oninput = (e) => oninput(e.target.value);
  row.appendChild(ta);
  return row;
}

function smallBtn(text, onclick, danger = false) {
  const b = el("button", "btn small" + (danger ? " danger" : ""), text);
  b.onclick = onclick;
  return b;
}

function downloadJson(filename, obj) {
  const blob = new Blob([JSON.stringify(obj, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

/* ---------- catalog ---------- */
export function renderCatalog({ catalogEl, qEl, kmetaEl, lab, onChange }) {
  const q = (qEl.value || "").trim().toLowerCase();
  catalogEl.innerHTML = "";

  const kernels = Object.values(lab.kernels).filter((k) => {
    if (!q) return true;
    const hay = `${k.id} ${k.name} ${(k.tags || []).join(" ")} ${k.summary || ""}`.toLowerCase();
    return hay.includes(q);
  });

  if (!kernels.length) {
    catalogEl.appendChild(el("div", "muted", "커널 없음"));
    kmetaEl.textContent = "Select a kernel";
    return;
  }

  if (!lab.selectedKernelId || !lab.kernels[lab.selectedKernelId]) {
    lab.selectedKernelId = kernels[0].id;
  }

  kernels.forEach((k) => {
    const card = el(
      "div",
      "card" + (k.id === lab.selectedKernelId ? " active" : "")
    );
    const top = el("div", "top");
    top.appendChild(el("div", "title", k.id));
    top.appendChild(
      el("div", "badge", `v${k.variants.length} · exp${k.experiments.length}`)
    );
    card.appendChild(top);
    card.appendChild(el("div", "sub", k.summary || "—"));

    card.onclick = () => {
      lab.selectedKernelId = k.id;
      lab.selectedBlockId = "overview";
      saveLab(lab);
      onChange();
    };
    catalogEl.appendChild(card);
  });

  const sel = lab.kernels[lab.selectedKernelId];
  kmetaEl.textContent = sel
    ? `${sel.id} · variants=${sel.variants.length} · suites=${sel.suites.length}`
    : "Select a kernel";
}

/* ---------- blocks & inspector ---------- */
export function renderBlocks({ blocksEl, inspectorEl, quickEl, lab, onChange }) {
  blocksEl.innerHTML = "";
  quickEl.querySelector(".panel").textContent = "—";

  const k = lab.kernels[lab.selectedKernelId];
  if (!k) return;

  const blocks = [
    "overview",
    "contract",
    "variants",
    "suites",
    "experiments",
    "measurements",
  ];

  blocks.forEach((id) => {
    const b = el(
      "div",
      "block" + (lab.selectedBlockId === id ? " active" : "")
    );
    b.appendChild(el("div", "h", id));
    b.onclick = () => {
      lab.selectedBlockId = id;
      saveLab(lab);
      onChange();
    };
    blocksEl.appendChild(b);
  });

  renderInspector({ inspectorEl, lab, k, onChange });
}

/* ---------- inspector ---------- */
function renderInspector({ inspectorEl, lab, k, onChange }) {
  inspectorEl.innerHTML = "";

  if (lab.selectedBlockId === "overview") {
    inspectorEl.appendChild(inputRow("name", k.name, (v) => (k.name = v)));
    inspectorEl.appendChild(
      textareaRow("summary", k.summary, (v) => (k.summary = v))
    );

    inspectorEl.appendChild(
      smallBtn("Seed bias_add template", () => {
        seedBiasAdd(lab);
        saveLab(lab);
        onChange();
      })
    );
  }

  saveLab(lab);
}

/* ---------- header buttons ---------- */
export function wireHeaderButtons({
  lab,
  btnNewKernel,
  btnExportKernel,
  btnImportKernel,
  btnReset,
  fileImportKernel,
  onChange,
}) {
  btnNewKernel.onclick = () => {
    newKernel(lab, "kernel");
    saveLab(lab);
    onChange();
  };

  btnExportKernel.onclick = () => {
    const k = lab.kernels[lab.selectedKernelId];
    if (!k) return alert("No kernel selected");
    downloadJson(`${k.id}.json`, k);
  };

  btnImportKernel.onclick = () => fileImportKernel.click();

  fileImportKernel.onchange = async () => {
    const f = fileImportKernel.files?.[0];
    if (!f) return;
    const k = JSON.parse(await f.text());
    if (!k?.id) return alert("Invalid kernel json");
    lab.kernels[k.id] = k;
    lab.selectedKernelId = k.id;
    lab.selectedBlockId = "overview";
    saveLab(lab);
    onChange();
    fileImportKernel.value = "";
  };

  btnReset.onclick = () => {
    localStorage.removeItem("rb_kernellab_v1");
    location.reload();
  };
}
