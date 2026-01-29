export const state = {
  nodes: [],
  selectedId: null,
  lastOutput: null,
  history: [],
};

const KEY = "recordbook_mvp_state_v2";

export function saveLocal(){
  localStorage.setItem(KEY, JSON.stringify({
    nodes: state.nodes,
    history: state.history,
  }));
}

export function loadLocal(){
  const raw = localStorage.getItem(KEY);
  if(!raw) return;
  try{
    const parsed = JSON.parse(raw);
    state.nodes = parsed.nodes || [];
    state.history = parsed.history || [];
  }catch{}
}

export function resetState(){
  state.nodes = [];
  state.selectedId = null;
  state.lastOutput = null;
}
