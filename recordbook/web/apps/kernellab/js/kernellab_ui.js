import * as KState from "./kernellab_state.js";
import { uid } from "../../graph/js/utils.js";

/* ---------- tiny DOM helpers ---------- */
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
  const text = JSON.stringify(obj, null, 2);
  const blob = new Blob([text], { type: "application/json" });
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

  const ids = Object.keys(lab.kernels);
  if (ids.length === 0) {
    const empty = el("div", "muted", "커널이 비어있음. New Kernel 또는 Import로 시작.");
    catalogEl.appendChild(empty);
    kmetaEl.textContent = "Select a kernel";
    return;
  }

  const list = ids
    .map((id) => lab.kernels[id])
    .filter((k) => {
      if (!q) return true;
      const hay = `${k.id} ${k.name} ${(k.tags || []).join(" ")} ${k.summary || ""}`.toLowerCase();
      return hay.includes(q);
    })
    .sort((a, b) => a.id.localeCompare(b.id));

  if (!lab.selectedKernelId || !lab.kernels[lab.selectedKernelId]) {
    lab.selectedKernelId = list[0]?.id || null;
  }

  list.forEach((k) => {
    const card = el("div", "card" + (k.id === lab.selectedKernelId ? " active" : ""));
    const top = el("div", "top");
    top.appendChild(el("div", "title", k.id));
    top.appendChild(
      el("div", "badge", `${k.kind || "op"} · v${(k.variants || []).length} · exp${(k.experiments || []).length}`)
    );
    card.appendChild(top);
    card.appendChild(el("div", "sub", k.summary || "—"));

    const tags = el("div", "kv");
    (k.tags || []).slice(0, 6).forEach((t) => tags.appendChild(el("span", "", t)));
    if ((k.tags || []).length) card.appendChild(tags);

    card.onclick = () => {
      lab.selectedKernelId = k.id;
      lab.selectedBlockId = "overview";
      KState.saveLab(lab);
      onChange();
    };
    catalogEl.appendChild(card);
  });

  const sel = lab.kernels[lab.selectedKernelId];
  kmetaEl.textContent = sel
    ? `${sel.id} · variants=${sel.variants.length} · suites=${sel.suites.length} · experiments=${sel.experiments.length}`
    : "Select a kernel";
}

/* ---------- middle blocks + quick view ---------- */
export function renderBlocks({ blocksEl, inspectorEl, quickEl, lab, onChange }) {
  blocksEl.innerHTML = "";
  quickEl.querySelector(".panel").textContent = "대표 결과/결론을 여기에 요약 표시";

  const kid = lab.selectedKernelId;
  if (!kid || !lab.kernels[kid]) {
    inspectorEl.classList.add("empty");
    inspectorEl.textContent = "왼쪽에서 커널 선택 후 블록을 클릭해.";
    return;
  }
  const k = lab.kernels[kid];

  const blocks = [
    { id: "overview", h: "Overview", sub: "한 줄 요약 + 태그 + 목적" },
    { id: "contract", h: "Contract / ABI", sub: "입출력 제약 + AttrBlob" },
    { id: "variants", h: "Variants", sub: "구현 후보 + eligibility(지원 조건)" },
    { id: "suites", h: "Benchmark Suites", sub: "측정 케이스 정의 (shape/dtype/params)" },
    { id: "experiments", h: "Experiments", sub: "변경점 ↔ 가설 ↔ 결과 ↔ 결론" },
    { id: "measurements", h: "Measurements", sub: "수치 기록 (latency/GB/s/링크)" }
  ];

  if (!lab.selectedBlockId) lab.selectedBlockId = "overview";

  blocks.forEach((b) => {
    const card = el("div", "block" + (b.id === lab.selectedBlockId ? " active" : ""));
    card.appendChild(el("div", "h", b.h));
    card.appendChild(el("div", "sub", b.sub));
    card.onclick = () => {
      lab.selectedBlockId = b.id;
      KState.saveLab(lab);
      onChange();
    };
    blocksEl.appendChild(card);
  });

  const lastExp = (k.experiments || [])[0];
  if (lastExp?.conclusion) {
    quickEl.querySelector(".panel").textContent = `Latest: ${lastExp.title} — ${lastExp.conclusion}`;
  } else {
    quickEl.querySelector(".panel").textContent = k.summary || "—";
  }

  renderInspector({ inspectorEl, lab, k, onChange });
}

/* ---------- inspector ---------- */
function renderInspector({ inspectorEl, lab, k, onChange }) {
  inspectorEl.classList.remove("empty");
  inspectorEl.innerHTML = "";

  const sec = lab.selectedBlockId || "overview";
  inspectorEl.appendChild(el("div", "muted", `kernel=${k.id} • section=${sec}`));

  if (sec === "overview") {
    inspectorEl.appendChild(inputRow("id", k.id, () => {}));
    inspectorEl.appendChild(inputRow("name", k.name, (v) => { k.name = v; KState.saveLab(lab); }));
    inspectorEl.appendChild(inputRow("kind", k.kind, (v) => { k.kind = v; KState.saveLab(lab); }));
    inspectorEl.appendChild(
      inputRow("tags (comma)", (k.tags || []).join(","), (v) => {
        k.tags = v.split(",").map((s) => s.trim()).filter(Boolean);
        KState.saveLab(lab);
      })
    );
    inspectorEl.appendChild(textareaRow("summary", k.summary, (v) => { k.summary = v; KState.saveLab(lab); }));
    inspectorEl.appendChild(textareaRow("notes", k.notes, (v) => { k.notes = v; KState.saveLab(lab); }));

    const row = el("div", "insp-row");
    row.appendChild(el("div", "muted", "템플릿은 data/kernels/*.json 로 관리. 필요하면 Import로 주입."));
    inspectorEl.appendChild(row);
  }

  if (sec === "contract") {
    inspectorEl.appendChild(textareaRow("contract (IO constraints)", k.contract, (v) => { k.contract = v; KState.saveLab(lab); }));
    inspectorEl.appendChild(textareaRow("ABI / AttrBlob", k.abi, (v) => { k.abi = v; KState.saveLab(lab); }));
  }

  if (sec === "variants") {
    const box = el("div", "box");
    box.appendChild(el("div", "t", "Variants"));

    (k.variants || []).forEach((v, idx) => {
      const vb = el("div", "box");
      vb.appendChild(el("div", "t", `#${idx} ${v.name}`));
      vb.appendChild(inputRow("name", v.name, (x) => { v.name = x; KState.saveLab(lab); }));
      vb.appendChild(inputRow("priority", String(v.priority ?? 0), (x) => { v.priority = Number(x) || 0; KState.saveLab(lab); }));
      vb.appendChild(inputRow("flags", String(v.flags ?? 0), (x) => { v.flags = Number(x) || 0; KState.saveLab(lab); }));
      vb.appendChild(textareaRow("supported (eligibility)", v.supported, (x) => { v.supported = x; KState.saveLab(lab); }));
      vb.appendChild(textareaRow("notes", v.notes, (x) => { v.notes = x; KState.saveLab(lab); }));

      vb.appendChild(
        smallBtn("Delete variant", () => {
          k.variants = (k.variants || []).filter((z) => z.id !== v.id);
          KState.saveLab(lab);
          onChange();
        }, true)
      );

      box.appendChild(vb);
    });

    box.appendChild(
      smallBtn("Add variant", () => {
        k.variants.unshift({ id: uid("v"), name: "new_variant", priority: 0, flags: 0, supported: "", notes: "" });
        KState.saveLab(lab);
        onChange();
      })
    );

    inspectorEl.appendChild(box);
  }

  if (sec === "suites") {
    const box = el("div", "box");
    box.appendChild(el("div", "t", "Benchmark Suites"));

    (k.suites || []).forEach((s, idx) => {
      const sb = el("div", "box");
      sb.appendChild(el("div", "t", `#${idx} ${s.name}`));
      sb.appendChild(inputRow("name", s.name, (x) => { s.name = x; KState.saveLab(lab); }));
      sb.appendChild(textareaRow("desc", s.desc, (x) => { s.desc = x; KState.saveLab(lab); }));

      (s.cases || []).forEach((c, j) => {
        const cb = el("div", "box");
        cb.appendChild(el("div", "t", `case#${j} ${c.label}`));
        cb.appendChild(inputRow("label", c.label, (x) => { c.label = x; KState.saveLab(lab); }));
        cb.appendChild(inputRow("shape", c.shape, (x) => { c.shape = x; KState.saveLab(lab); }));
        cb.appendChild(inputRow("dtype", c.dtype, (x) => { c.dtype = x; KState.saveLab(lab); }));
        cb.appendChild(inputRow("params", c.params, (x) => { c.params = x; KState.saveLab(lab); }));

        cb.appendChild(
          smallBtn("Delete case", () => {
            s.cases = (s.cases || []).filter((z) => z.id !== c.id);
            KState.saveLab(lab);
            onChange();
          }, true)
        );

        sb.appendChild(cb);
      });

      sb.appendChild(
        smallBtn("Add case", () => {
          s.cases = s.cases || [];
          s.cases.push({ id: uid("case"), label: "new_case", shape: "", dtype: "", params: "" });
          KState.saveLab(lab);
          onChange();
        })
      );

      sb.appendChild(
        smallBtn("Delete suite", () => {
          k.suites = (k.suites || []).filter((z) => z.id !== s.id);
          KState.saveLab(lab);
          onChange();
        }, true)
      );

      box.appendChild(sb);
    });

    box.appendChild(
      smallBtn("Add suite", () => {
        k.suites.unshift({ id: uid("suite"), name: "new_suite", desc: "", cases: [] });
        KState.saveLab(lab);
        onChange();
      })
    );

    inspectorEl.appendChild(box);
  }

  if (sec === "experiments") {
    const box = el("div", "box");
    box.appendChild(el("div", "t", "Experiments (optimization log)"));

    (k.experiments || []).forEach((e, idx) => {
      const eb = el("div", "box");
      eb.appendChild(el("div", "t", `#${idx} ${e.title}`));
      eb.appendChild(inputRow("title", e.title, (x) => { e.title = x; KState.saveLab(lab); }));
      eb.appendChild(inputRow("suite_id (optional)", e.suite_id || "", (x) => { e.suite_id = x; KState.saveLab(lab); }));
      eb.appendChild(textareaRow("what changed", e.change, (x) => { e.change = x; KState.saveLab(lab); }));
      eb.appendChild(textareaRow("why", e.why, (x) => { e.why = x; KState.saveLab(lab); }));
      eb.appendChild(textareaRow("hypothesis", e.hypothesis, (x) => { e.hypothesis = x; KState.saveLab(lab); }));
      eb.appendChild(textareaRow("result (numbers/summary)", e.result, (x) => { e.result = x; KState.saveLab(lab); }));
      eb.appendChild(textareaRow("proof (ncu/nsys links + key metrics)", e.proof, (x) => { e.proof = x; KState.saveLab(lab); }));
      eb.appendChild(textareaRow("conclusion", e.conclusion, (x) => { e.conclusion = x; KState.saveLab(lab); }));
      eb.appendChild(textareaRow("next", e.next, (x) => { e.next = x; KState.saveLab(lab); }));

      eb.appendChild(
        smallBtn("Delete experiment", () => {
          k.experiments = (k.experiments || []).filter((z) => z.id !== e.id);
          KState.saveLab(lab);
          onChange();
        }, true)
      );

      box.appendChild(eb);
    });

    box.appendChild(
      smallBtn("Add experiment", () => {
        k.experiments.unshift({
          id: uid("exp"),
          ts: Date.now(),
          title: `exp_${new Date().toLocaleString()}`,
          suite_id: "",
          change: "",
          why: "",
          hypothesis: "",
          result: "",
          proof: "",
          conclusion: "",
          next: ""
        });
        KState.saveLab(lab);
        onChange();
      })
    );

    inspectorEl.appendChild(box);
  }

  if (sec === "measurements") {
    const box = el("div", "box");
    box.appendChild(el("div", "t", "Measurements (manual)"));

    (k.measurements || []).slice(0, 12).forEach((m, idx) => {
      const mb = el("div", "box");
      mb.appendChild(el("div", "t", `#${idx} ${new Date(m.ts || Date.now()).toLocaleString()}`));

      mb.appendChild(inputRow("suite_id", m.suite_id || "", (x) => { m.suite_id = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("case_id", m.case_id || "", (x) => { m.case_id = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("variant_id", m.variant_id || "", (x) => { m.variant_id = x; KState.saveLab(lab); }));

      mb.appendChild(inputRow("device", m.device || "", (x) => { m.device = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("cuda", m.cuda || "", (x) => { m.cuda = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("driver", m.driver || "", (x) => { m.driver = x; KState.saveLab(lab); }));

      mb.appendChild(inputRow("dtype", m.dtype || "", (x) => { m.dtype = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("shape", m.shape || "", (x) => { m.shape = x; KState.saveLab(lab); }));

      mb.appendChild(inputRow("lat_us", String(m.lat_us || ""), (x) => { m.lat_us = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("GB/s", String(m.gb_s || ""), (x) => { m.gb_s = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("TFLOP/s", String(m.tflops || ""), (x) => { m.tflops = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("speedup", String(m.speedup || ""), (x) => { m.speedup = x; KState.saveLab(lab); }));

      mb.appendChild(inputRow("ncu link/path", m.ncu || "", (x) => { m.ncu = x; KState.saveLab(lab); }));
      mb.appendChild(inputRow("nsys link/path", m.nsys || "", (x) => { m.nsys = x; KState.saveLab(lab); }));
      mb.appendChild(textareaRow("notes", m.notes || "", (x) => { m.notes = x; KState.saveLab(lab); }));

      mb.appendChild(
        smallBtn("Delete measurement", () => {
          k.measurements = (k.measurements || []).filter((z) => z.id !== m.id);
          KState.saveLab(lab);
          onChange();
        }, true)
      );

      box.appendChild(mb);
    });

    box.appendChild(
      smallBtn("Add measurement", () => {
        k.measurements.unshift({
          id: uid("m"),
          ts: Date.now(),
          suite_id: "",
          case_id: "",
          variant_id: "",
          device: "",
          driver: "",
          cuda: "",
          dtype: "",
          shape: "",
          lat_us: "",
          gb_s: "",
          tflops: "",
          speedup: "",
          notes: "",
          ncu: "",
          nsys: ""
        });
        KState.saveLab(lab);
        onChange();
      })
    );

    inspectorEl.appendChild(box);
  }
}

/* ---------- header buttons (kernel-level export/import) ---------- */
export function wireHeaderButtons({
  lab,
  btnNewKernel,
  btnExportKernel,
  btnImportKernel,
  btnReset,
  fileImportKernel,
  onChange
}) {
  btnNewKernel.onclick = () => {
    KState.newKernel(lab, "kernel");
    KState.saveLab(lab);
    onChange();
  };

  btnReset.onclick = () => {
    localStorage.removeItem("rb_kernellab_v1");
    location.reload();
  };

  btnExportKernel.onclick = () => {
    const kid = lab.selectedKernelId;
    const k = kid ? lab.kernels[kid] : null;
    if (!k) return alert("No kernel selected.");
    downloadJson(`${k.id}.json`, k);
  };

  btnImportKernel.onclick = () => fileImportKernel.click();

  fileImportKernel.onchange = async () => {
    const f = fileImportKernel.files?.[0];
    if (!f) return;

    const txt = await f.text();
    try {
      const kid = KState.importKernelJsonText(lab, txt);
      KState.saveLab(lab);
      onChange();
      alert(`Imported: ${kid}`);
    } catch (e) {
      alert(String(e?.message || "Invalid JSON."));
    } finally {
      fileImportKernel.value = "";
    }
  };
}
