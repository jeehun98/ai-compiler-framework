export function clearWires(svgEl){
  svgEl.innerHTML = "";
}

export function setWiresDim(svgEl, isDim){
  svgEl.querySelectorAll(".wire").forEach(p => p.classList.toggle("dim", !!isDim));
}

function ensureSvgSize(workspaceEl, svgEl){
  const w = workspaceEl.clientWidth;
  const h = workspaceEl.clientHeight;
  svgEl.setAttribute("viewBox", `0 0 ${w} ${h}`);
}

function portCenterInWorkspace(workspaceEl, portEl){
  const wRect = workspaceEl.getBoundingClientRect();
  const pRect = portEl.getBoundingClientRect();
  const x = (pRect.left + pRect.width/2) - wRect.left;
  const y = (pRect.top  + pRect.height/2) - wRect.top;
  return { x, y };
}

function cubicPath(a, b){
  const dx = Math.max(30, Math.abs(b.x - a.x) * 0.45);
  const c1 = { x: a.x + dx, y: a.y };
  const c2 = { x: b.x - dx, y: b.y };
  return `M ${a.x} ${a.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${b.x} ${b.y}`;
}

export function drawWires(workspaceEl, svgEl, nodesLayerEl){
  clearWires(svgEl);
  ensureSvgSize(workspaceEl, svgEl);

  const nodes = Array.from(nodesLayerEl.querySelectorAll(".node"));
  if(nodes.length <= 1) return;

  for(let i=0;i<nodes.length-1;i++){
    const aNode = nodes[i];
    const bNode = nodes[i+1];

    const outPort = aNode.querySelector(".port.out");
    const inPort  = bNode.querySelector(".port.in");
    if(!outPort || !inPort) continue;

    const a = portCenterInWorkspace(workspaceEl, outPort);
    const b = portCenterInWorkspace(workspaceEl, inPort);

    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.setAttribute("d", cubicPath(a, b));
    path.setAttribute("class", "wire");
    svgEl.appendChild(path);
  }
}
