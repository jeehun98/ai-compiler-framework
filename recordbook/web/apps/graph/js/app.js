import { getEls, initTabs } from "./dom.js";
import { state, loadLocal, saveLocal, resetState } from "./state.js";
import { mountPalette } from "./palette.js";
import { mountWorkspace, renderPipelinePreview } from "./workspace.js";
import { mountInspector } from "./inspector.js";
import { renderOutput } from "./output.js";
import { renderHistory } from "./history.js";
import { drawWires, setWiresDim, clearWires } from "./wires.js";
import { uid, clone } from "./utils.js";

const els = getEls();

function rerender({save=true} = {}){
  mountWorkspace(els.nodesLayer, {
    onChange: () => rerender(),
    onDragStart: () => setWiresDim(els.wires, true),
    onDragEnd: () => {
      setWiresDim(els.wires, false);
      requestAnimationFrame(draw);
    },
  });

  mountInspector(els.inspector, { onChange: () => rerender() });
  renderPipelinePreview(els.pipelinePreview);
  renderOutput(els.output);
  renderHistory(els.history, { onSelect: () => rerender({save:false}) });

  requestAnimationFrame(draw);

  if(save) saveLocal();
}

function draw(){
  const hasNodes = els.nodesLayer.querySelectorAll(".node").length > 0;
  if(!hasNodes){
    clearWires(els.wires);
    return;
  }
  drawWires(els.workspace, els.wires, els.nodesLayer);
}

function mockCompile(){
  const ops = [];
  const trace = [];

  state.nodes.forEach((n) => {
    if(n.op === "Linear"){
      ops.push({
        kind:"gemm_epilogue",
        kid:"gemm_bias_relu_f16_tc_wmma_out_f16_v0",
        note:`mnk=(64,${n.params.out},${n.params.in}) transB=True (mock)`
      });
      trace.push({ kid:"gemm_bias_relu_f16_tc_wmma_out_f16_v0", kind:"gemm_epilogue" });
    }else if(n.op === "ReLU"){
      ops.push({ kind:"(fused?)", kid:"-", note:`ReLU might be fused (mock)` });
    }else if(n.op === "MseGrad"){
      ops.push({ kind:"mse_grad", kid:"mse_grad_f16_v0", note:"out_matches_in0=True (mock)" });
      trace.push({ kid:"mse_grad_f16_v0", kind:"mse_grad" });
    }else if(n.op === "ReduceSum"){
      ops.push({
        kind:"reduce_sum",
        kid:"reduce_sum_keep_lastdim_f16_v0",
        note:`axis=${n.params.axis} keepdim=${n.params.keepdim} (mock)`
      });
      trace.push({ kid:"reduce_sum_keep_lastdim_f16_v0", kind:"reduce_sum" });
    }else if(n.op === "SgdStep"){
      ops.push({ kind:"sgd_step", kid:"sgd_step_f16_half2_v0", note:`lr=${n.params.lr} (mock)` });
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
    nodesSnapshot: clone(state.nodes),
  };
  state.history.unshift(run);
  state.history = state.history.slice(0, 30);
}

function wireButtons(){
  els.btnNew.onclick = () => { resetState(); rerender(); };
  els.btnSave.onclick = () => { saveLocal(); alert("Saved (localStorage)."); };
  els.btnLoad.onclick = () => { loadLocal(); rerender({save:false}); alert("Loaded."); };
  els.btnCompile.onclick = () => { mockCompile(); rerender(); };
}

function boot(){
  initTabs();
  mountPalette(els.palette, () => rerender());

  wireButtons();
  loadLocal();
  rerender({save:false});

  els.workspace.addEventListener("scroll", () => requestAnimationFrame(draw));
  window.addEventListener("resize", () => requestAnimationFrame(draw));
}
boot();
