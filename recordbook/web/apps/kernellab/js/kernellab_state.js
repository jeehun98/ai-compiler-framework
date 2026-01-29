import { uid } from "../../graph/js/utils.js";

const LS_KEY = "rb_kernellab_v1";

export function defaultLab(){
  return {
    version: 1,
    selectedKernelId: null,
    selectedBlockId: null,
    // kernelId -> kernel
    kernels: {}
  };
}

export function loadLab(){
  try{
    const raw = localStorage.getItem(LS_KEY);
    if(!raw) return defaultLab();
    const v = JSON.parse(raw);
    if(!v || typeof v !== "object") return defaultLab();
    // minimal guards
    if(!v.kernels) v.kernels = {};
    if(v.version !== 1) v.version = 1;
    return v;
  }catch{
    return defaultLab();
  }
}

export function saveLab(lab){
  localStorage.setItem(LS_KEY, JSON.stringify(lab));
}

export function resetLab(){
  localStorage.removeItem(LS_KEY);
}

export function ensureKernel(lab, kernelId){
  if(!lab.kernels[kernelId]){
    lab.kernels[kernelId] = {
      id: kernelId,
      name: kernelId,
      kind: "op",
      tags: [],
      summary: "—",
      contract: "",
      abi: "",
      notes: "",
      variants: [
        // {id,name,priority,flags,supported,notes}
      ],
      suites: [
        // {id,name,desc,cases:[{id,label,shape,dtype,params}]}
      ],
      experiments: [
        // {id,title,ts,change,why,hypothesis,suite_id,result,proof,conclusion,next}
      ],
      measurements: [
        // {id,ts,suite_id,case_id,variant_id,device,driver,cuda,dtype,shape,lat_us,gb_s,tflops,speedup,notes,ncu,nsys}
      ],
    };
  }
  return lab.kernels[kernelId];
}

export function newKernel(lab, baseId="kernel"){
  const id = uid(baseId);
  ensureKernel(lab, id);
  lab.kernels[id].name = id;
  lab.selectedKernelId = id;
  lab.selectedBlockId = "overview";
  return id;
}

export function seedBiasAdd(lab){
  const id = "bias_add";
  const k = ensureKernel(lab, id);
  k.name = "bias_add";
  k.kind = "elementwise";
  k.tags = ["cuda", "broadcast", "memory-bound"];
  k.summary = "Out = Y + bias (broadcast on last dim). vec2(half2) fastpath when N even + 4B aligned.";
  k.contract = "Inputs: Y(rank>=2, contiguous), bias(1D, len=N). Output same shape as Y.";
  k.abi = "AttrBlob schema 0 or 'BADD' (int64 axis). axis only last-dim (-1 or rank-1).";

  if(k.variants.length === 0){
    k.variants.push(
      { id: uid("v"), name: "bias_add_f32", priority: 0, flags: 0, supported: "f32, contiguous, axis last, bias len==N", notes: "" },
      { id: uid("v"), name: "bias_add_f16_naive", priority: 0, flags: 0, supported: "f16, contiguous, axis last, bias len==N", notes: "" },
      { id: uid("v"), name: "bias_add_f16_vec2_half2", priority: 10, flags: 0, supported: "f16 + N even + Y/B/O 4B aligned", notes: "half2 path" },
    );
  }

  if(k.suites.length === 0){
    k.suites.push({
      id: uid("suite"),
      name: "shape_sweep_basic",
      desc: "M,N sweep for realistic shapes",
      cases: [
        { id: uid("case"), label: "M=4096 N=768 f16", shape: "M=4096,N=768", dtype: "f16", params: "axis=-1" },
        { id: uid("case"), label: "M=4096 N=767 f16 (odd N)", shape: "M=4096,N=767", dtype: "f16", params: "axis=-1 (force fallback)" },
        { id: uid("case"), label: "M=16384 N=3072 f16", shape: "M=16384,N=3072", dtype: "f16", params: "axis=-1" },
      ]
    });
  }

  lab.selectedKernelId = id;
  lab.selectedBlockId = "overview";
  return id;
}
