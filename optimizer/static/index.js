document.getElementById("rmi1-layer1").selectedIndex = 2;
document.getElementById("rmi1-layer2").selectedIndex = 0;
document.getElementById("rmi1-bf").selectedIndex = 0;
document.getElementById("cdf1-dataset").selectedIndex = 0;

// https://stackoverflow.com/questions/10420352/converting-file-size-in-bytes-to-human-readable-string/10420404
function humanFileSize(size) {
    if (size < 1024) return size + ' B';
    let i = Math.floor(Math.log(size) / Math.log(1024));
    let num = (size / Math.pow(1024, i));
    let round = Math.round(num);
    num = round < 10 ? num.toFixed(2) : round < 100 ? num.toFixed(1) : round;
    return `${num} ${'KMGTPEZY'[i-1]}B`;
}


const g = new CDFGraph("cdf-view", "info-table-1");
g.reset();
g.update();

const t = new MeasureTable("rmi-table");
t.clear();

document.getElementById("cdf1-dataset").onchange = function(e) {
    const dataset = e.target.options[e.target.selectedIndex].value;
    g.dataset = dataset;
    t.dataset = dataset;
    
    g.reset();
    g.update();

    t.clear();
};
document.getElementById("cdf-reset").onclick = function() { g.reset(); g.update(); };

function updateDisplay() {
    const l1 = document.getElementById("rmi1-layer1");
    const layer1 = l1.options[l1.selectedIndex].value;
    const l2 = document.getElementById("rmi1-layer2");
    const layer2 = l2.options[l2.selectedIndex].value;

    const bf = document.getElementById("rmi1-bf");
    const branchingFactor = bf.options[bf.selectedIndex].value;

    g.setLayers(`${layer1},${layer2}`);
    g.setBranchingFactor(parseInt(branchingFactor));
    g.update();
}



document.getElementById("rmi1-layer1").onchange = updateDisplay;
document.getElementById("rmi1-layer2").onchange = updateDisplay;
document.getElementById("rmi1-bf").onchange = updateDisplay;



document.getElementById("rmi1-save").onclick = function() {
    if (!g.currentStats) return;
    t.addRMI(g.currentStats);
};

