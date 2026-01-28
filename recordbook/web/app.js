const PALETTE = [
  { op: "Linear",    tag: "nn",   defaults: { in: 8, out: 8, bias: true, dtype: "float16" } },
  { op: "ReLU",      tag: "act",  defaults: { inplace: false } },
  { op: "MseGrad",   tag: "loss", defaults: { reduction: "mean" } },
  { op: "ReduceSum", tag: "red",  defaults: { axis: 0, keepdim: false } },
  { op: "SgdStep",   tag: "opt",  defaults: { lr: 0.01 } },
];

let state = {
  nodes: [],           // {id, op, params}
  selectedId: null,
  lastOutput: null,    // mock compile result
  history: [],         // {id, name, ts, output, nodesSnapshot}
};

const els = {
  palette: document.getElementById("palette"),
  workspace: document.getElementById("workspace"),
  inspector: document.getElementById("inspector"),
  pipelinePreview: document.getElementById("pipelinePreview"),
  history: document.getElementById("history"),
  btnNew: document.getElementById("btnNew"),
  btnSave: document.getElementById("btnSave"),
  btnLoad: document.getElementById("btnLoad"),
  btnCompile: document.getElementById("btnCompile"),
  output: document.getElementById("output"),
};

function uid(prefix="n"){
  return `${prefix}_${Math.random().toString(16).slice(2)}_${Date.now().toString(16)}`;
}

function saveLocal(){
  localStorage.setItem("recordbook_mvp_state", JSON.stringify({
    nodes: state.nodes,
    history: state.history,
  }));
}

function loadLocal(){
  const raw = localStorage.getItem("recordbook_mvp_state");
  if(!raw) return;
  try{
    const parsed = JSON.parse(raw);
    state.nodes = parsed.nodes || [];
    state.history = parsed.history || [];
  }catch(e){}
}

function reset(){
  state.nodes = [];
  state.selectedId = null;
  state.lastOutput = null;
  renderAll();
}

function initPalette(){
  els.palette.innerHTML = "";
  PALETTE.forEach(item => {
    const btn = document.createElement("div");
    btn.className = "block-btn";
    btn.innerHTML = `<div>${item.op}</div><div class="block-tag">${item.tag}</div>`;
    btn.onclick = () => addNode(item.op, item.defaults);
    els.palette.appendChild(btn);
  });
}

function addNode(op, defaults){
  const node = { id: uid("node"), op, params: structuredClone(defaults || {}) };
  state.nodes.push(node);
  state.selectedId = node.id;
  renderAll();
}

function removeNode(id){
  state.nodes = state.nodes.filter(n => n.id !== id);
  if(state.selectedId === id) state.selectedId = null;
  renderAll();
}

function moveNode(fromIdx, toIdx){
  if(toIdx < 0 || toIdx >= state.nodes.length) return;
  const a = state.nodes;
  const [x] = a.splice(fromIdx, 1);
  a.splice(toIdx, 0, x);
}

function renderWorkspace(){
  els.workspace.innerHTML = "";

  if(state.nodes.length === 0){
    const empty = document.createElement("div");
    empty.style.color = "#9aa8bb";
    empty.style.padding = "10px";
    empty.textContent = "왼쪽 Blocks에서 추가해.";
    els.workspace.appendChild(empty);
    return;
  }

  state.nodes.forEach((n, idx) => {
    const div = document.createElement("div");
    div.className = "node" + (state.selectedId === n.id ? " selected" : "");
    div.setAttribute("draggable", "true");
    div.dataset.id = n.id;
    div.dataset.idx = idx;

    const desc = summarizeParams(n);
    div.innerHTML = `
      <div class="left">
        <div class="op">${n.op}</div>
        <div class="desc">${desc}</div>
      </div>
      <div class="right">
        <span class="pill">#${idx}</span>
        <button class="icon-btn" data-act="up">↑</button>
        <button class="icon-btn" data-act="down">↓</button>
        <button class="icon-btn" data-act="del">✕</button>
      </div>
    `;

    div.onclick = (e) => {
      const act = e.target?.dataset?.act;
      if(act === "del") return removeNode(n.id);
      if(act === "up")  { moveNode(idx, idx-1); renderAll(); return; }
      if(act === "down"){ moveNode(idx, idx+1); renderAll(); return; }
      state.selectedId = n.id;
      renderAll();
    };

    // Drag & drop
    div.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", String(idx));
      e.dataTransfer.effectAllowed = "move";
    });
    div.addEventListener("dragover", (e) => {
      e.preventDefault();
      e.dataTransfer.dropEffect = "move";
    });
    div.addEventListener("drop", (e) => {
      e.preventDefault();
      const from = Number(e.dataTransfer.getData("text/plain"));
      const to = idx;
      if(Number.isFinite(from) && Number.isFinite(to) && from !== to){
        moveNode(from, to);
        renderAll();
      }
    });

    els.workspace.appendChild(div);
  });
}

function summarizeParams(n){
  const p = n.params || {};
  if(n.op === "Linear") return `in=${p.in}, out=${p.out}, bias=${p.bias}, dtype=${p.dtype}`;
  if(n.op === "ReLU") return `inplace=${p.inplace}`;
  if(n.op === "MseGrad") return `reduction=${p.reduction}`;
  if(n.op === "ReduceSum") return `axis=${p.axis}, keepdim=${p.keepdim}`;
  if(n.op === "SgdStep") return `lr=${p.lr}`;
  return Object.keys(p).length ? JSON.stringify(p) : "-";
}

function renderInspector(){
  const sel = state.nodes.find(n => n.id === state.selectedId);
  if(!sel){
    els.inspector.classList.add("empty");
    els.inspector.textContent = "블럭을 선택해.";
    return;
  }
  els.inspector.classList.remove("empty");
  els.inspector.innerHTML = `
    <div class="badge"><span>Selected</span><span class="mono">${sel.op}</span></div>
    <div class="mono" style="margin-top:8px; color:#9aa8bb;">id: ${sel.id}</div>
    <div id="form"></div>
  `;

  const form = els.inspector.querySelector("#form");
  form.innerHTML = "";

  // simple form builder per op
  if(sel.op === "Linear"){
    form.appendChild(inputRow("in", sel.params.in, "number", v => sel.params.in = Number(v)));
    form.appendChild(inputRow("out", sel.params.out, "number", v => sel.params.out = Number(v)));
    form.appendChild(selectRow("bias", String(sel.params.bias), ["true","false"], v => sel.params.bias = (v==="true")));
    form.appendChild(selectRow("dtype", sel.params.dtype, ["float16","float32"], v => sel.params.dtype = v));
  }else if(sel.op === "ReLU"){
    form.appendChild(selectRow("inplace", String(sel.params.inplace), ["false","true"], v => sel.params.inplace = (v==="true")));
  }else if(sel.op === "MseGrad"){
    form.appendChild(selectRow("reduction", sel.params.reduction, ["mean","sum","none"], v => sel.params.reduction = v));
  }else if(sel.op === "ReduceSum"){
    form.appendChild(inputRow("axis", sel.params.axis, "number", v => sel.params.axis = Number(v)));
    form.appendChild(selectRow("keepdim", String(sel.params.keepdim), ["false","true"], v => sel.params.keepdim = (v==="true")));
  }else if(sel.op === "SgdStep"){
    form.appendChild(inputRow("lr", sel.params.lr, "number", v => sel.params.lr = Number(v)));
  }

  // rerender preview text on change
  form.querySelectorAll("input,select").forEach(el => {
    el.addEventListener("change", () => renderAll(false));
    el.addEventListener("input", () => renderAll(false));
  });
}

function inputRow(label, value, type, onChange){
  const row = document.createElement("div");
  row.className = "form-row";
  row.innerHTML = `
    <label>${label}</label>
    <input type="${type}" value="${value}" />
  `;
  const inp = row.querySelector("input");
  inp.onchange = () => onChange(inp.value);
  inp.oninput = () => onChange(inp.value);
  return row;
}

function selectRow(label, value, options, onChange){
  const row = document.createElement("div");
  row.className = "form-row";
  const opts = options.map(o => `<option value="${o}" ${o===value?"selected":""}>${o}</option>`).join("");
  row.innerHTML = `
    <label>${label}</label>
    <select>${opts}</select>
  `;
  const sel = row.querySelector("select");
  sel.onchange = () => onChange(sel.value);
  return row;
}

function renderPipelinePreview(){
  if(state.nodes.length === 0){
    els.pipelinePreview.textContent = "Empty";
    return;
  }
  const s = state.nodes.map((n,i) => `${i}:${n.op}`).join("  →  ");
  els.pipelinePreview.textContent = s;
}

function initTabs(){
  document.querySelectorAll(".tab").forEach(t => {
    t.onclick = () => {
      document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
      t.classList.add("active");
      const tab = t.dataset.tab;
      document.querySelectorAll(".panel").forEach(p => {
        p.classList.toggle("hidden", p.dataset.panel !== tab);
      });
    };
  });
}

function mockCompile(){
  // “스크래치 조립 결과가 lowered/trace로 보이는 느낌”만 주면 됨
  const ops = [];
  const trace = [];

  state.nodes.forEach((n) => {
    if(n.op === "Linear"){
      ops.push({ kind:"gemm_epilogue", kid:"gemm_bias_relu_f16_tc_wmma_out_f16_v0", note:`mnk=(64,${n.params.out},${n.params.in}) transB=True` });
      trace.push({ kid:"gemm_bias_relu_f16_tc_wmma_out_f16_v0", kind:"gemm_epilogue" });
    }else if(n.op === "ReLU"){
      ops.push({ kind:"(fused?)", kid:"-", note:`ReLU might be fused` });
    }else if(n.op === "MseGrad"){
      ops.push({ kind:"mse_grad", kid:"mse_grad_f16_v0", note:"out_matches_in0=True" });
      trace.push({ kid:"mse_grad_f16_v0", kind:"mse_grad" });
    }else if(n.op === "ReduceSum"){
      ops.push({ kind:"reduce_sum", kid:"reduce_sum_keep_lastdim_f16_v0", note:`axis=${n.params.axis} keepdim=${n.params.keepdim}` });
      trace.push({ kid:"reduce_sum_keep_lastdim_f16_v0", kind:"reduce_sum" });
    }else if(n.op === "SgdStep"){
      ops.push({ kind:"sgd_step", kid:"sgd_step_f16_half2_v0", note:`lr=${n.params.lr}` });
      trace.push({ kid:"sgd_step_f16_half2_v0", kind:"sgd_step" });
    }
  });

  const out = {
    created_at: new Date().toISOString(),
    summary: {
      blocks: state.nodes.length,
      lowered_ops: ops.length,
      trace_len: trace.length,
      kid_version: "2026-01-20 (mock)"
    },
    lowered_ops: ops,
    trace
  };
  state.lastOutput = out;

  const run = {
    id: uid("run"),
    name: `run_${new Date().toLocaleString()}`,
    ts: Date.now(),
    output: out,
    nodesSnapshot: structuredClone(state.nodes),
  };
  state.history.unshift(run);
  state.history = state.history.slice(0, 30);
  saveLocal();
  renderOutput();
  renderHistory();
}

function renderOutput(){
  const out = state.lastOutput;
  const pSummary = els.output.querySelector('[data-panel="summary"]');
  const pLowered = els.output.querySelector('[data-panel="lowered"]');
  const pTrace   = els.output.querySelector('[data-panel="trace"]');

  if(!out){
    pSummary.innerHTML = `<div class="mono" style="color:#9aa8bb;">No output yet. Compile을 눌러.</div>`;
    pLowered.innerHTML = "";
    pTrace.innerHTML = "";
    return;
  }

  pSummary.innerHTML = `
    <div class="kv"><div>kid_version</div><div class="mono">${out.summary.kid_version}</div></div>
    <div class="kv"><div>blocks</div><div class="mono">${out.summary.blocks}</div></div>
    <div class="kv"><div>lowered_ops</div><div class="mono">${out.summary.lowered_ops}</div></div>
    <div class="kv"><div>trace_len</div><div class="mono">${out.summary.trace_len}</div></div>
    <div class="kv"><div>created_at</div><div class="mono">${out.created_at}</div></div>
  `;

  pLowered.innerHTML = out.lowered_ops.map((op, i) => `
    <div class="kv">
      <div class="mono">#${String(i).padStart(2,"0")} ${op.kind}</div>
      <div class="mono">${op.kid}</div>
    </div>
    <div class="mono" style="color:#9aa8bb; margin:6px 0 10px 0;">${op.note}</div>
  `).join("");

  pTrace.innerHTML = out.trace.map((t, i) => `
    <div class="kv">
      <div class="mono">${String(i).padStart(2,"0")} ${t.kind}</div>
      <div class="mono">${t.kid}</div>
    </div>
  `).join("");
}

function renderHistory(){
  els.history.innerHTML = "";
  if(state.history.length === 0){
    const d = document.createElement("div");
    d.style.color = "#9aa8bb";
    d.style.padding = "8px";
    d.textContent = "No runs yet.";
    els.history.appendChild(d);
    return;
  }

  state.history.forEach(item => {
    const div = document.createElement("div");
    div.className = "history-item";
    div.innerHTML = `
      <div class="name">${item.name}</div>
      <div class="meta mono">blocks=${item.output.summary.blocks} lowered_ops=${item.output.summary.lowered_ops} trace=${item.output.summary.trace_len}</div>
    `;
    div.onclick = () => {
      state.nodes = structuredClone(item.nodesSnapshot);
      state.selectedId = null;
      state.lastOutput = item.output;
      renderAll();
    };
    els.history.appendChild(div);
  });
}

function renderAll(save=true){
  renderWorkspace();
  renderInspector();
  renderPipelinePreview();
  renderOutput();
  renderHistory();
  if(save) saveLocal();
}

function wireButtons(){
  els.btnNew.onclick = () => reset();
  els.btnSave.onclick = () => { saveLocal(); alert("Saved (localStorage)."); };
  els.btnLoad.onclick = () => { loadLocal(); renderAll(false); alert("Loaded."); };
  els.btnCompile.onclick = () => mockCompile();
}

function boot(){
  initPalette();
  initTabs();
  wireButtons();
  loadLocal();
  renderAll(false);
}
boot();
