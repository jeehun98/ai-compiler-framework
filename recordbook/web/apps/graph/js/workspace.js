import { state } from "./state.js";

export function summarizeParams(n){
  const p = n.params || {};
  if(n.op === "Linear") return `in=${p.in}, out=${p.out}, bias=${p.bias}, dtype=${p.dtype}`;
  if(n.op === "ReLU") return `inplace=${p.inplace}`;
  if(n.op === "MseGrad") return `reduction=${p.reduction}`;
  if(n.op === "ReduceSum") return `axis=${p.axis}, keepdim=${p.keepdim}`;
  if(n.op === "SgdStep") return `lr=${p.lr}`;
  return Object.keys(p).length ? JSON.stringify(p) : "-";
}

export function renderPipelinePreview(pipelineEl){
  if(state.nodes.length === 0){
    pipelineEl.textContent = "Empty";
    return;
  }
  pipelineEl.textContent = state.nodes.map((n,i)=>`${i}:${n.op}`).join("  →  ");
}

export function mountWorkspace(nodesLayerEl, {
  onChange,
  onDragStart,
  onDragEnd
}){
  nodesLayerEl.innerHTML = "";

  if(state.nodes.length === 0){
    const empty = document.createElement("div");
    empty.style.color = "#9aa8bb";
    empty.style.padding = "10px";
    empty.textContent = "왼쪽 Blocks에서 추가해.";
    nodesLayerEl.appendChild(empty);
    return;
  }

  state.nodes.forEach((n, idx) => {
    const div = document.createElement("div");
    div.className = "node" + (state.selectedId === n.id ? " selected" : "");
    div.setAttribute("draggable", "true");
    div.dataset.id = n.id;
    div.dataset.idx = idx;

    div.innerHTML = `
      <span class="port in" data-port="in"></span>
      <span class="port out" data-port="out"></span>

      <div class="left">
        <div class="op">${n.op}</div>
        <div class="desc">${summarizeParams(n)}</div>
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
      if(act === "del"){
        state.nodes = state.nodes.filter(x => x.id !== n.id);
        if(state.selectedId === n.id) state.selectedId = null;
        onChange?.();
        return;
      }
      if(act === "up"){ moveNode(idx, idx-1); onChange?.(); return; }
      if(act === "down"){ moveNode(idx, idx+1); onChange?.(); return; }

      state.selectedId = n.id;
      onChange?.();
    };

    div.addEventListener("dragstart", (e) => {
      e.dataTransfer.setData("text/plain", String(idx));
      e.dataTransfer.effectAllowed = "move";
      onDragStart?.();
    });

    div.addEventListener("dragend", () => {
      onDragEnd?.();
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
        onChange?.();
      }
    });

    nodesLayerEl.appendChild(div);
  });
}

function moveNode(fromIdx, toIdx){
  if(toIdx < 0 || toIdx >= state.nodes.length) return;
  const a = state.nodes;
  const [x] = a.splice(fromIdx, 1);
  a.splice(toIdx, 0, x);
}
