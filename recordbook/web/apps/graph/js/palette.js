import { state } from "./state.js";
import { uid, clone } from "./utils.js";

export const PALETTE = [
  { op:"Linear", tag:"nn", defaults:{ in:8, out:8, bias:true, dtype:"float16" } },
  { op:"ReLU", tag:"act", defaults:{ inplace:false } },
  { op:"MseGrad", tag:"loss", defaults:{ reduction:"mean" } },
  { op:"ReduceSum", tag:"red", defaults:{ axis:0, keepdim:false } },
  { op:"SgdStep", tag:"opt", defaults:{ lr:0.01 } },
];

export function mountPalette(paletteEl, onChange){
  paletteEl.innerHTML = "";
  PALETTE.forEach(item => {
    const btn = document.createElement("div");
    btn.className = "block-btn";
    btn.innerHTML = `<div>${item.op}</div><div class="block-tag">${item.tag}</div>`;
    btn.onclick = () => {
      const node = { id: uid("node"), op:item.op, params: clone(item.defaults) };
      state.nodes.push(node);
      state.selectedId = node.id;
      onChange?.();
    };
    paletteEl.appendChild(btn);
  });
}
