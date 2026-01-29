import { state } from "./state.js";

export function mountInspector(inspectorEl, { onChange }){
  const sel = state.nodes.find(n => n.id === state.selectedId);
  if(!sel){
    inspectorEl.classList.add("empty");
    inspectorEl.textContent = "블럭을 선택해.";
    return;
  }

  inspectorEl.classList.remove("empty");
  inspectorEl.innerHTML = `
    <div class="badge"><span>Selected</span><span class="mono">${sel.op}</span></div>
    <div class="mono" style="margin-top:8px; color:#9aa8bb;">id: ${sel.id}</div>
    <div id="form"></div>
  `;

  const form = inspectorEl.querySelector("#form");
  form.innerHTML = "";

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

  form.querySelectorAll("input,select").forEach(el => {
    el.addEventListener("change", () => onChange?.());
    el.addEventListener("input", () => onChange?.());
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
