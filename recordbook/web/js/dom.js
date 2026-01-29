export function getEls(){
  return {
    palette: document.getElementById("palette"),
    history: document.getElementById("history"),

    workspace: document.getElementById("workspace"),
    wires: document.getElementById("wires"),
    nodesLayer: document.getElementById("nodesLayer"),
    pipelinePreview: document.getElementById("pipelinePreview"),

    inspector: document.getElementById("inspector"),
    output: document.getElementById("output"),

    btnNew: document.getElementById("btnNew"),
    btnSave: document.getElementById("btnSave"),
    btnLoad: document.getElementById("btnLoad"),
    btnCompile: document.getElementById("btnCompile"),
  };
}

export function initTabs(){
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
