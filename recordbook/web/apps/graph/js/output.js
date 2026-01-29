import { state } from "./state.js";

export function renderOutput(outputEl){
  const out = state.lastOutput;
  const pSummary = outputEl.querySelector('[data-panel="summary"]');
  const pLowered = outputEl.querySelector('[data-panel="lowered"]');
  const pTrace   = outputEl.querySelector('[data-panel="trace"]');

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
