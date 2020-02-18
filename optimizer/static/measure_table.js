const BS_TIME = 764.326;

const BTREE_TIME = {
    "osm_cellids_200M_uint64": 526.425,
    "fb_200M_uint64": 519.719,
    "books_200M_uint64": 532.601,
    "lognormal_200M_uint64": 519.245,
    "normal_200M_uint64": 515.904
};

class ParetoPlot {
    constructor (elID, bsTime, btTime, btSize) {
        this.elID = elID;
        const root = d3.select(`#${elID}`);
        this.wgraph = 780;
        this.hgraph = 480;
        
        this.paddingLeft = 85;
        this.paddingBottom = 70;
        this.paddingRight = 25;
        this.paddingTop = 10;

        this.tooltip = d3.select("#pareto-tooltip");

        root.selectAll("*").remove();
        this.svg = root
            .append("svg")
            .attr("width", this.wgraph)
            .attr("height", this.hgraph);

        this.xaxis = this.svg.append('g').attr('class', "xaxis");
        this.yaxis = this.svg.append('g').attr('class', "yaxis");
        this.labels = this.svg.append('g').attr('class', "labels");
        this.front = this.svg.append('path');
        this.points = this.svg.append('g').attr('class', "pareto-point");

        
        this.data = [{
            "type": "BS", id: "BS",
            "latency": bsTime,
            "size": 0
        }, {
            "type": "BT", id: "BT",
            "latency": btTime,
            "size": btSize
        }];
        
        this.xLabel = this.svg.append('text')
            .attr("class", "axis-label")
            .text("Size")
            .attr("x", this.paddingLeft + ((this.wgraph - this.paddingLeft) / 2))
            .attr("y", this.hgraph - 22)
            .attr("text-anchor", "middle");

        this.yLabel = this.svg.append('text')
            .attr("transform", `rotate(-90)`)
            .attr("class", "axis-label")
            .text("Latency (ns)")
            .attr("y", 20)
            .attr("x", 0 - ((this.hgraph - this.paddingBottom) / 2))
            .attr("text-anchor", "middle");

        this.update();
    }

    idForRMI(stats) {
        const id = `${stats["layers"]} ${stats["branching factor"]}`;
        return id;
    }
    
    addRMI(stats) {
        if (!("latency" in stats))
            return;
        
        this.data.push({
            "type": "RMI",
            "id": this.idForRMI(stats),
            "latency": stats["latency"],
            "size": stats["size binary search"]
        });
        this.update();
    }

    deleteRMI(stats) {
        console.log(this.idForRMI(stats));
        console.log(this.data);
        this.data = this.data.filter(d => d.id != this.idForRMI(stats));
        console.log(this.data);
        this.update();
    }

    update() {
        this.createScales();
        this.addAxes();
        this.plotData();
    }

    createScales() {
        this.xMin = Math.min(...this.data.map(d => d.size)) + 1;
        this.xMax = Math.max(...this.data.map(d => d.size)) * 1.05;
        this.yMin = 0; //Math.min(...this.data.map(d => d.latency));
        this.yMax = Math.max(...this.data.map(d => d.latency)) * 1.05;
        
        this.xScale = d3.scaleSqrt()
            .domain([this.xMin, this.xMax])
            .range([
                this.paddingLeft,
                this.wgraph - this.paddingRight
            ]);
        this.yScale = d3.scaleLinear()
            .domain([this.yMin, this.yMax])
            .range([
                this.hgraph - this.paddingBottom,
                this.paddingTop
            ]);
    }

    addAxes() {
        const nticks = 8;
        const xAxis = d3.axisBottom()
              .scale(this.xScale)
              .tickValues([1, 8*1024*1024, 80*1024*1024, 200*1024*1024, 400*1024*1024])
              .tickFormat(d => humanFileSize(d));
        
        const yAxis = d3.axisLeft()
              .scale(this.yScale)
              .ticks(nticks);

        this.xaxis
            .attr("transform", "translate(0," + (this.hgraph - this.paddingBottom) + ")")
            .call(xAxis);

        this.yaxis
            .attr("transform", "translate(" + this.paddingLeft + ",0)")
            .call(yAxis);
    }

    colorForPoint(pt) {
        if (pt.type == "RMI")
            return "yellow";

        if (pt.type == "BT")
            return "blue";

        if (pt.type == "BS")
            return "red";

        return "gray";
    }

    onFront(pt) {
        for (let other of this.data) {
            if (other.id != pt.id
                && other.size <= pt.size
                && other.latency <= pt.latency)
                return false;
        }

        return true;
    }

    opacityForPoint(pt) {
        return this.onFront(pt) ? 1.0 : 0.5;
    }

    strokeForPoint(pt) {
        return this.onFront(pt) ? "black" : "gray";
    }

    tooltipHTML(stats) {
        let size = stats.size;
        let latency = stats.latency;
        let label = "";
        console.log(stats);
        switch(stats.type) {
        case "BS":
            label = "Binary search";
            break;
        case "BT":
            label = "B-Tree";
            break;
        case "RMI":
            label = stats.id;
            break;
        }
        return `
<center><strong>${label}</strong></center>
Latency: ${latency} ns<br/>
Size: ${humanFileSize(size)}
`;
    }
    
    plotData() {
        const points = this.points
              .selectAll("circle")
              .data(this.data);

        points.enter().append("circle")
            .merge(points)
            .attr("cx", d => this.xScale(d.size + 1))
            .attr("cy", d => this.yScale(d.latency))
            .attr("r", 7)
            .style("fill", d => this.colorForPoint(d))
            .style("stroke", d => this.strokeForPoint(d))
            .style("stroke-width", 2)
            .style("opacity", d => this.opacityForPoint(d))
            .on("mouseover", d => {
                console.log(d);
                this.tooltip.html(this.tooltipHTML(d))
                    .style("display", "block")
                    .style("left", (d3.event.pageX - 60) + "px")
                    .style("top", (d3.event.pageY + 20) + "px");
            }).on("mouseout", () => {
                this.tooltip.style("display", "none");
            });

        points.exit().remove();

        const line = d3.line()
              .x(d => this.xScale(d.size + 1))
              .y(d => this.yScale(d.latency));

        const frontPoints = this.data.filter(d => this.onFront(d))
              .sort((a, b) => b.latency - a.latency);
        frontPoints.push({"latency": frontPoints[frontPoints.length-1].latency,
                          "size": this.xMax});
        console.log(frontPoints);
        this.front
            .style("fill", "none")
            .style("stroke", "black")
            .style("stroke-opacity", "0.75")
            .style("stroke-width", "3px")
            .datum(frontPoints)
            .attr("d", line);

    }
}


class MeasureTable {
    constructor(elID) {
        this.rmis = [];
        this.dataset = "osm_cellids_200M_uint64";

        const root = d3.select(`#${elID}`).append("div");
        const table = root.append("table").attr("class", "table table-hover");
        const headerRow = table.append("thead")
              .append("tr");
        headerRow.append("th").text("Layers");
        headerRow.append("th").text("Branching factor");
        headerRow.append("th").text("Max error");
        headerRow.append("th").text("Average error");
        headerRow.append("th").text("Size");
        headerRow.append("th").text("Latency");
        headerRow.append("th").text("");
        headerRow.append("th").text("");


        const bsRow = table.append("tbody").append("tr");
        bsRow.append("td").attr("colspan", 4).text("Binary Search").style("text-align", "center");
        bsRow.append("td").text("0 KB");
        bsRow.append("td").text(BS_TIME);
        bsRow.append("td");
        bsRow.append("td");

        const btRow = table.append("tbody").append("tr");
        btRow.append("td").attr("colspan", 4).text("B-Tree").style("text-align", "center");
        btRow.append("td").text("400 MB");
        this.btLatencyCell = btRow.append("td").text(BTREE_TIME[this.dataset]);
        btRow.append("td");
        btRow.append("td");
        
        this.tooltips = root.append("div").attr("id", "table-tooltips").style("display", "none");
        this.tbody = table.append("tbody");
        this.inspector = new Inspector();    
    }

    clear() {
        this.rmis = [];
        this.paretoPlot = new ParetoPlot("pareto-plot", BS_TIME, BTREE_TIME[this.dataset], 400 * 1024 * 1024);
        this.update();
    }

    async getLatency(stats) {
        if (!this.dataset) return;

        const resp = await fetch(`/measure/${this.dataset}/${stats["layers"]}/${stats["branching factor"]}`);
        const json = (await resp.json());

        const latency = json.result[0];

        for (let storedRMI of this.rmis) {
            if (storedRMI["layers"] == stats["layers"]
                && storedRMI["branching factor"] == stats["branching factor"]) {
                storedRMI["latency"] = latency;
                this.paretoPlot.addRMI(storedRMI);
                break;
            }
        }

        this.update();
    }
    
    addRMI(stats) {
        for (let rmi of this.rmis) {
            if (rmi["layers"] == stats["layers"]
                && rmi["branching factor"] == stats["branching factor"])
                return;
        }
        this.rmis.unshift(stats);
        this.update();
        this.getLatency(stats);
    }

    deleteRow(idx) {
        this.paretoPlot.deleteRMI(this.rmis[idx]);
        this.rmis.splice(idx, 1);
        this.update();
    }

    inspectRow(idx) {
        const stats = this.rmis[idx];
        const color = this.latencyToClass(stats);
        const bst = BS_TIME;
        const bt = BTREE_TIME[this.dataset];

        this.inspector.show(stats, this.dataset, color, bst, bt);
    }

    onFront(stats) {
        if (!("latency" in stats))
            return true;

        for (let other of this.rmis) {
            if (!("latency" in other))
                continue;

            if (other["namespace"] != stats["namespace"]
                && other["size binary search"] <= stats["size binary search"]
                && other["latency"] <= stats["latency"])
                return false;
        }

        return true;
    }
    
    latencyToClass(stats) {
        if ("latency" in stats) {
            if (!this.onFront(stats)) {
                return "table-secondary";
            }
            const latency = stats.latency;
            if (latency > BS_TIME)
                return "table-danger";
            
            if (latency > BTREE_TIME[this.dataset])
                return "table-warning";

            return "table-success";
        }
        return "";
    }

    tooltip(stats) {
        const root = d3.select(document.createElement("div"));
        if (!("latency" in stats)) {
            root.text("Please wait...");
            return root;
        }

        const summary = root.append("div");
        switch (this.latencyToClass(stats)) {
        case "table-success": {
            summary.html("This RMI is <strong>faster</strong> than both binary search and a B-Tree!");
            break;
        }
        case "table-warning": {
            summary.html("This RMI is <strong>faster</strong> than binary search, but <strong>slower</strong> than a B-Tree.");
            break;
        }
        case "table-danger":
            summary.html("This RMI is <strong>slower</strong> than binary search!");
            break;
        }

        root.append("br");
        const lstats = root.append("table").attr("class", "table-dark");
        const header = lstats.append("thead").append("tr");
        header.append("th").text("");
        header.append("th").text("Binary search");
        header.append("th").text("B-Tree");
        header.append("th").text("RMI");

        const body = lstats.append("tbody");
        const latencyRow = body.append("tr");
        latencyRow.append("td").text("Latency");
        latencyRow.append("td").text(`${BS_TIME.toFixed(0)}ns`);
        latencyRow.append("td").text(`${BTREE_TIME[this.dataset].toFixed(0)}ns`);
        latencyRow.append("td").text(`${stats.latency.toFixed(0)}ns`);

        const sizeRow = body.append("tr");
        sizeRow.append("td").text("Size");
        sizeRow.append("td").text("0 KB");
        sizeRow.append("td").text("400 MB");
        sizeRow.append("td").text(humanFileSize(stats["size binary search"]));

        
        return root;
    }
    
    update() {
        const rows = this.tbody
              .selectAll("tr")
              .data(this.rmis);

        const allRows = rows.enter()
              .append("tr")
              .merge(rows)
              .attr("class", d => this.latencyToClass(d));


        this.tooltips.selectAll("div").remove();
        const tooltips = this.tooltips
              .selectAll("div")
              .data(this.rmis);

        tooltips.enter().append("div")
            .merge(tooltips)
            .attr("id", (_, idx) => `tooltip-row-${idx}`)
            .append(d => this.tooltip(d).node());

        const cells = allRows.selectAll("td")
              .data((d, idx) => {
                  return [
                      d["layers"], d["branching factor"],
                      d["max error"].toFixed(2),
                      d["average error"].toFixed(2),
                      humanFileSize(d["size binary search"]),
                      ("latency" in d ? d["latency"] : "waiting..."),
                      {"idx": idx,
                       "text": '<i class="fas fa-trash-alt"></i>',
                       "fn": () => this.deleteRow(idx)},
                      {"idx": idx,
                       "text": '<i class="fas fa-info-circle"></i>',
                       "fn": () => this.inspectRow(idx)}
                  ];
              });
        
        cells.enter()
            .append("td")
            .merge(cells)
            .html((d, idx) => {
                if (d == "waiting...") {
                    return '<div class="spinner-border" role="status"><span class="sr-only">Loading...</span></div>';
                }
                if (typeof d === "object")
                    return d.text;
                return d;
            }).on("click", (d) => {
                if (typeof d !== "object") return;
                d.fn();
            });

        cells.exit().remove();
        rows.exit().remove();

           
    }
}
