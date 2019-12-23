const g = new CDFGraph("cdf-view");
g.reset();
g.update();

document.getElementById("cdf1-dataset").onchange = function(e) {
    const dataset = e.target.options[e.target.selectedIndex].value;
    g.dataset = dataset;
    g.reset();
    g.update(); // TODO show progress, await
};
document.getElementById("cdf-reset").onclick = function() { g.reset(); g.update(); };

document.getElementById("rmi1-update").onclick = function() {
    const l1 = document.getElementById("rmi1-layer1");
    const layer1 = l1.options[l1.selectedIndex].value;
    const l2 = document.getElementById("rmi1-layer2");
    const layer2 = l2.options[l2.selectedIndex].value;

    const bf = document.getElementById("rmi1-bf");
    const branchingFactor = bf.options[bf.selectedIndex].value;

    g.layers = `${layer1},${layer2}`;
    g.bf = parseInt(branchingFactor);
    g.update();
};
