// apps/kernellab/js/kernellab_state.js
import { uid } from "../../graph/js/utils.js";

console.log("[KLab] kernellab_state.js LOADED", import.meta.url);

const LS_KEY = "rb_kernellab_v1";

/* ----------------------------- lab default ----------------------------- */
export function defaultLab() {
  return {
    version: 1,
    selectedKernelId: null,
    selectedBlockId: "overview",
    kernels: {}, // kernelId -> kernel
  };
}

/* ----------------------------- tiny helpers ---------------------------- */
function asText(x) {
  // UI textarea는 string만 받게 강제
  if (Array.isArray(x)) return x.join("\n");
  if (x == null) return "";
  return String(x);
}

function asArrayOfStrings(x) {
  // tags 같은 곳: string[] 강제
  if (Array.isArray(x)) return x.map((v) => String(v)).filter((s) => s.length > 0);
  if (typeof x === "string") {
    // "a,b,c" 들어오면 대충 split도 가능하지만 여기선 보수적으로 빈 배열 처리
    return x
      .split(",")
      .map((s) => s.trim())
      .filter(Boolean);
  }
  return [];
}

function asNumber(x, fallback = 0) {
  const n = Number(x);
  return Number.isFinite(n) ? n : fallback;
}

/* ----------------------------- normalize ------------------------------ */
function normalizeKernel(k) {
  // 최소 필수 필드 보정 (import kernel json이 와도 UI가 안 깨지게)
  if (!k || typeof k !== "object") return null;
  if (!k.id) return null;

  // top-level scalar fields
  if (k.name == null) k.name = k.id;
  if (k.kind == null) k.kind = "op";
  k.tags = asArrayOfStrings(k.tags);

  if (k.summary == null) k.summary = "—";

  // IMPORTANT: allow array or string in JSON, but store as text for UI
  k.contract = asText(k.contract);
  k.abi = asText(k.abi);
  k.notes = asText(k.notes);

  // collections
  if (!Array.isArray(k.variants)) k.variants = [];
  if (!Array.isArray(k.suites)) k.suites = [];
  if (!Array.isArray(k.experiments)) k.experiments = [];
  if (!Array.isArray(k.measurements)) k.measurements = [];

  // Variants normalize
  k.variants = k.variants
    .filter((v) => v && typeof v === "object")
    .map((v) => {
      if (!v.id) v.id = uid("v");
      if (v.name == null) v.name = "variant";
      v.priority = asNumber(v.priority, 0);
      v.flags = asNumber(v.flags, 0);

      // allow supported as string[] or string -> store as text
      v.supported = asText(v.supported);
      v.notes = asText(v.notes);

      return v;
    });

  // Suites normalize
  k.suites = k.suites
    .filter((s) => s && typeof s === "object")
    .map((s) => {
      if (!s.id) s.id = uid("suite");
      if (s.name == null) s.name = "suite";
      s.desc = asText(s.desc);

      if (!Array.isArray(s.cases)) s.cases = [];
      s.cases = s.cases
        .filter((c) => c && typeof c === "object")
        .map((c) => {
          if (!c.id) c.id = uid("case");
          if (c.label == null) c.label = "case";
          c.shape = asText(c.shape);
          c.dtype = asText(c.dtype);
          c.params = asText(c.params);
          return c;
        });

      return s;
    });

  // Experiments normalize
  k.experiments = k.experiments
    .filter((e) => e && typeof e === "object")
    .map((e) => {
      if (!e.id) e.id = uid("exp");
      // ts==0(템플릿) 허용. null/undefined일 때만 채움
      if (e.ts == null) e.ts = Date.now();
      if (e.title == null) e.title = "exp";
      e.suite_id = asText(e.suite_id);

      e.change = asText(e.change);
      e.why = asText(e.why);
      e.hypothesis = asText(e.hypothesis);
      e.result = asText(e.result);
      e.proof = asText(e.proof);
      e.conclusion = asText(e.conclusion);
      e.next = asText(e.next);

      return e;
    });

  // Measurements normalize
  k.measurements = k.measurements
    .filter((m) => m && typeof m === "object")
    .map((m) => {
      if (!m.id) m.id = uid("m");
      // ts==0(템플릿) 허용. null/undefined일 때만 채움
      if (m.ts == null) m.ts = Date.now();

      m.suite_id = asText(m.suite_id);
      m.case_id = asText(m.case_id);
      m.variant_id = asText(m.variant_id);

      m.device = asText(m.device);
      m.driver = asText(m.driver);
      m.cuda = asText(m.cuda);

      m.dtype = asText(m.dtype);
      m.shape = asText(m.shape);

      m.lat_us = asText(m.lat_us);
      m.gb_s = asText(m.gb_s);
      m.tflops = asText(m.tflops);
      m.speedup = asText(m.speedup);

      m.notes = asText(m.notes);
      m.ncu = asText(m.ncu);
      m.nsys = asText(m.nsys);

      return m;
    });

  return k;
}

/* ----------------------------- storage ------------------------------- */
export function loadLab() {
  try {
    const raw = localStorage.getItem(LS_KEY);
    if (!raw) return defaultLab();

    const v = JSON.parse(raw);
    if (!v || typeof v !== "object") return defaultLab();

    if (v.version !== 1) v.version = 1;
    if (!v.kernels || typeof v.kernels !== "object") v.kernels = {};

    // kernels normalize
    for (const [kid, k] of Object.entries(v.kernels)) {
      const nk = normalizeKernel(k);
      if (!nk) {
        delete v.kernels[kid];
        continue;
      }
      // key와 id mismatch 정리
      if (nk.id !== kid) {
        delete v.kernels[kid];
        v.kernels[nk.id] = nk;
      } else {
        v.kernels[kid] = nk;
      }
    }

    // selection normalize
    if (v.selectedKernelId && !v.kernels[v.selectedKernelId]) {
      v.selectedKernelId = Object.keys(v.kernels)[0] || null;
    }
    if (!v.selectedBlockId) v.selectedBlockId = "overview";

    return v;
  } catch {
    return defaultLab();
  }
}

export function saveLab(lab) {
  localStorage.setItem(LS_KEY, JSON.stringify(lab));
}

export function resetLab() {
  localStorage.removeItem(LS_KEY);
}

/* ----------------------------- kernel ops ---------------------------- */
export function ensureKernel(lab, kernelId) {
  if (!lab.kernels[kernelId]) {
    lab.kernels[kernelId] = normalizeKernel({
      id: kernelId,
      name: kernelId,
      kind: "op",
      tags: [],
      summary: "—",
      contract: "",
      abi: "",
      notes: "",
      variants: [],
      suites: [],
      experiments: [],
      measurements: [],
    });
  } else {
    lab.kernels[kernelId] = normalizeKernel(lab.kernels[kernelId]);
  }
  return lab.kernels[kernelId];
}

export function newKernel(lab, baseId = "kernel") {
  const id = uid(baseId);
  ensureKernel(lab, id);
  lab.kernels[id].name = id;
  lab.selectedKernelId = id;
  lab.selectedBlockId = "overview";
  return id;
}

/* ------------------------ library(json) loading ------------------------ */
// load a single kernel json via fetch
export async function loadKernelJson(path) {
  const r = await fetch(path, { cache: "no-store" });
  if (!r.ok) throw new Error(`fetch failed: ${path} (${r.status})`);
  const obj = await r.json();
  return obj;
}

// upsert one kernel object into lab (with normalization)
export function upsertKernel(lab, kernelObj) {
  const nk = normalizeKernel(kernelObj);
  if (!nk) return false;
  lab.kernels[nk.id] = nk;
  return true;
}

// import from file text (used by Import button)
export function importKernelJsonText(lab, txt) {
  const obj = JSON.parse(txt);
  const ok = upsertKernel(lab, obj);
  if (!ok) throw new Error("Invalid kernel JSON (missing id or wrong type).");
  lab.selectedKernelId = obj.id;
  lab.selectedBlockId = "overview";
  return obj.id;
}
