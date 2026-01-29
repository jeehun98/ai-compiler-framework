import { state } from "./state.js";
import { clone } from "./utils.js";

export function renderHistory(historyEl, { onSelect }){
  historyEl.innerHTML = "";

  if(state.history.length === 0){
    const d = document.createElement("div");
    d.style.color = "#9aa8bb";
    d.style.padding = "8px";
    d.textContent = "No runs yet.";
    historyEl.appendChild(d);
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
      state.nodes = clone(item.nodesSnapshot);
      state.selectedId = null;
      state.lastOutput = item.output;
      onSelect?.();
    };
    historyEl.appendChild(div);
  });
}
