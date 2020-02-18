class ErrorGraph {
    constructor (elID) {
        this.elID = elID;
        const root = d3.select(`#${elID}`);
        this.wgraph = 578;
        this.hgraph = 432;
        this.layers = "linear,linear";
        this.bf = "1024";
        
        this.paddingLeft = 85;
        this.paddingBottom = 70;
        this.paddingRight = 25;
        this.paddingTop = 10;

        root.selectAll("*").remove();
        this.svg = root
            .append("svg")
            .attr("width", this.wgraph)
            .attr("height", this.hgraph);

        this.xaxis = this.svg.append('g').attr('class', "xaxis");
        this.yaxis = this.svg.append('g').attr('class', "yaxis");
        this.labels = this.svg.append('g').attr('class', "labels");
        this.rmiPoints = this.svg.append('path').attr('class', "line");
        
        this.xLabel = this.svg.append('text')
            .attr("class", "axis-label")
            .text("% of data")
            .attr("x", this.paddingLeft + ((this.wgraph - this.paddingLeft) / 2))
            .attr("y", this.hgraph - 22)
            .attr("text-anchor", "middle");

        this.yLabel = this.svg.append('text')
            .attr("transform", `rotate(-90)`)
            .attr("class", "axis-label")
            .text("Error (# of keys)")
            .attr("y", 20)
            .attr("x", 0 - ((this.hgraph - this.paddingBottom) / 2))
            .attr("text-anchor", "middle");
    }

    setLayers(layers) {
        this.layers = layers;
    }

    setBranchingFactor(bf) {
        this.bf = bf;
    }

    async update() {
        //showProgressIndicator();
        const resp = await fetch(`/variance/${this.dataset}/${this.layers}/${this.bf}`);
        const json = (await resp.json()).results;
        const len = json.length + 0.0;
        this.data = json.map((d, i) => ({ "x": (i+1.0)/len, "y": d }));
        this.xMin = 0;
        this.xMax = 1;

        console.log(this.data);
        this.yMin = this.data[0].y;
        this.yMax = this.data[this.data.length - 1].y;
        
        this.createScales();
        this.addAxes();
        this.plotData();
    }

    createScales() {
        this.xScale = d3.scaleLinear()
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
        const offsetXScale = d3.scaleLinear()
              .domain([0, this.xMax - this.xMin])
              .range([ this.paddingLeft, this.wgraph - this.paddingRight ]);

        const offsetYScale = d3.scaleLinear()
              .domain([0, this.yMax - this.yMin])
              .range([ this.hgraph - this.paddingBottom, this.paddingTop ] );
        
        const nticks = 8;
        const xAxis = d3.axisBottom()
              .scale(offsetXScale)
              .ticks(nticks, "%");
        const yAxis = d3.axisLeft()
              .scale(offsetYScale)
              .ticks(nticks, "s");

        this.xaxis
            .attr("transform", "translate(0," + (this.hgraph - this.paddingBottom) + ")")
            .call(xAxis);

        this.yaxis
            .attr("transform", "translate(" + this.paddingLeft + ",0)")
            .call(yAxis);
    }

    plotData() {
        const line = d3.line()
              .x(d => this.xScale(d.x))
              .y(d => this.yScale(d.y));


        this.rmiPoints
            .style("fill", "none")
            .style("stroke", "black")
            .style("stroke-opacity", "1")
            .style("stroke-width", "3px")
            .datum(this.data)
            .attr("d", line);
    }
}

class Inspector {
    constructor() {
        const inspector = d3.select("#rmi-inspect-content");
        this.inspectorTitle = d3.select("#rmi-inspect-label");
        this.inspectorText = d3.select("#rmi-inspect-status");
        inspector.append("br");
        const inspectorTable = inspector.append("table").attr("class", "table");
        const inspectorTableHeaders = inspectorTable.append("thead").append("tr");
        inspectorTableHeaders.append("th").text("");
        inspectorTableHeaders.append("th").text("Binary search");
        inspectorTableHeaders.append("th").text("B Tree");
        inspectorTableHeaders.append("th").text("RMI");
        this.inspectorTBody = inspectorTable.append("tbody");
    }


    show(stats, dataset, color, bst, bt) {
        this.dataset = dataset;
        this.inspectorTitle.text(`Inspecting ${stats["layers"]} ${stats["branching factor"]}`);
        const lbl = this.inspectorText;
        switch (color) {
        case "table-success": {
            lbl.html(`<div class="alert alert-success">This RMI is <strong>faster</strong> than both binary search and a B-Tree!</div>`);
            break;
        }
        case "table-warning": {
            lbl.html(`<div class="alert alert-warning">This RMI is <strong>faster</strong> than binary search, but <strong>slower</strong> than a B-Tree.</div>`);
            break;
        }
        case "table-danger":
            lbl.html(`<div class="alert alert-danger">This RMI is <strong>slower</strong> than binary search!</div>`);
            break;
        case "table-secondary":
            lbl.html(`<div class="alert alert-secondary">This RMI is strictly worse (bigger and slower) than another RMI in the table.</div>`);
            break;

        }


        const tclass = "text-" + color.substring(6);
        const tdata = [
            ["<strong>Latency<strong>",
             `${bst.toFixed(0)} ns`,
             `${bt.toFixed(0)} ns`,
             `<div class="${tclass}">${stats.latency.toFixed(0)} ns</div>`],
            ["<strong>Size</strong>",
             "0 KB", "400 MB", humanFileSize(stats["size binary search"])]
        ];
        
        const rows = this.inspectorTBody.selectAll("tr")
              .data(tdata);
        
        const allRows = rows.enter()
              .append("tr")
              .merge(rows);

        console.log(allRows);

        const cells = allRows.selectAll("td")
              .data((d, idx) => d);
        

        cells.enter().append("td")
            .merge(cells)
            .html(d => d);
        
        const graph = new ErrorGraph("rmi-inspect-graph");
        graph.dataset = this.dataset;
        graph.setLayers(stats["layers"]);
        graph.setBranchingFactor(stats["branching factor"]);
        graph.update();

        
        $("#rmi-inspect").modal("show");
    }
}
