function closest(arr, val) {
    const metric = v => Math.abs(v.x - val);
    return arr.reduce((min, val) => metric(min) < metric(val) ? min : val, arr[0]);
}

class CDFGraph {
    constructor(elID) {
        this.elID = elID;
        this.wgraph = 640;
        this.hgraph = 480;
        this.dataset = "osm_cellids_200M_uint64";
        this.layers = "linear,linear";
        this.bf = "1024";
        
        this.paddingLeft = 100;
        this.paddingBottom = 70;
        this.paddingRight = 10;
        this.paddingTop = 10;
        
        this.paddingBar = 2;

        this.svg = d3.select(`#${this.elID}`)
            .append("svg")
            .attr("width", this.wgraph)
            .attr("height", this.hgraph);

        this.xaxis = this.svg.append('g').attr('class', "xaxis");
        this.yaxis = this.svg.append('g').attr('class', "yaxis");
        this.labels = this.svg.append('g').attr('class', "labels");
        this.cdfDots = this.svg.append('g').attr("class", "dots");
        this.rmiDots = this.svg.append('g').attr("class", "dots");
        this.cdfPoints = this.svg.append('path').attr('class', "line");
        this.rmiPoints = this.svg.append('path').attr('class', "line");

        this.hoverLine = this.svg.append("line").attr("class", "hover-line");
        this.hoverCircleRMI = this.svg.append("circle").attr("class", "hover-circle");
        this.hoverCircleData = this.svg.append("circle").attr("class", "hover-circle");
        
        this.selectorRect = this.svg.append("rect").attr("class", "selector-rect");
        
        this.reset();
        
        this.xLabel = this.svg.append('text')
            .attr("class", "axis-label")
            .text("Key")
            .attr("x", this.paddingLeft + ((this.wgraph - this.paddingLeft) / 2))
            .attr("y", this.hgraph - 22)
            .attr("text-anchor", "middle");

        this.yLabel = this.svg.append('text')
            .attr("transform", `rotate(-90)`)
            .attr("class", "axis-label")
            .text("Position")
            .attr("y", 20)
            .attr("x", 0 - ((this.hgraph - this.paddingBottom) / 2))
            .attr("text-anchor", "middle");

        d3.drag()
            .on("start", d => {
                this.initRect(d3.event.y);
            })
            .on("drag", d => {
                this.updateRect(d3.event.y);
            })
            .on("end", d => {
                this.finalizeRect();
            })(this.svg);

        this.svg.on("mouseleave", d => {
            this.hoverLine.style("display", "none");
            this.hoverCircleData.style("display", "none");
            this.hoverCircleRMI.style("display", "none");
        });
        this.svg.on("mousemove", d => { this.updateHover(d3.event.x); });

    }

    reset(x) {
        this.setRange(0, 200000000);
    }

    initRect(y) {
        this.selectorRect
            .attr("x", this.xScale(this.xMin))
            .attr("y", y)
            .attr("starty", y)
            .attr("width", this.wgraph - this.paddingLeft)
            .attr("height", 0)
            .style("display", "");
    }
    
    updateRect(y) {
        const y1 = this.selectorRect.attr("starty");
        const y2 = y;

        let ty = Math.max(this.yScale(this.yMax), Math.min(y1, y2));
        let by = Math.min(this.yScale(this.yMin), Math.max(y1, y2));

        this.selectorRect
            .attr("y", ty)
            .attr("height", by - ty);
    }

    finalizeRect() {
        const y1 = parseInt(this.selectorRect.attr("y"));
        const y2 = y1 + parseInt(this.selectorRect.attr("height"));

        const y1val = Math.ceil(this.yScale.invert(y1));
        const y2val = Math.floor(this.yScale.invert(y2));
                
        this.selectorRect.style("display", "none");


        this.setRange(y2val, y1val);
        this.update();
    }

    updateHover(x) {
        this.hoverLine
            .attr("x1", x)
            .attr("x2", x)
            .attr("y1", this.yScale(this.yMin))
            .attr("y2", this.yScale(this.yMax))
            .style("display", "");


        const invX = this.xScale.invert(x);
        const rmi = closest(this.rmiData, invX).y;
        const data = closest(this.cdfData, invX).y;
        this.hoverCircleData
            .attr("cx", x)
            .attr("cy", this.yScale(data))
            .attr("r", 4)
            .style("display", "");

        this.hoverCircleRMI
            .attr("cx", x)
            .attr("cy", this.yScale(rmi))
            .attr("r", 4)
            .style("fill", "red")
            .style("display", "");
        
        
    }

    setRange(min, max) {
        this.keyMin = min;
        this.keyMax = max;
    }

    async update() {
        const resp = await fetch(`/data/${this.dataset}/${this.keyMin}/${this.keyMax}`);
        const json = (await resp.json()).data;

        const respRMI = await fetch(`/rmi/${this.dataset}/${this.layers}/${this.bf}/${this.keyMin}/${this.keyMax}`);
        const jsonRMI = (await respRMI.json()).data;

        // y values for RMI may not be monotonic
        this.xMin = Math.min(json[0].x, jsonRMI[0].x);
        this.xMax = Math.max(json[json.length-1].x,
                             Math.max(...jsonRMI.map(d => d.x)));


        this.yMin = Math.min(json[0].y,
                             Math.min(...jsonRMI.map(d => d.y)));
        this.yMax = Math.max(json[json.length-1].y,
                             Math.max(...jsonRMI.map(d => d.y)));
        
        this.cdfData = json;
        this.rmiData = jsonRMI;
        
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
        
        const nticks = 5;
        const xAxis = d3.axisBottom()
              .scale(offsetXScale)
              .ticks(nticks, "s");
        const yAxis = d3.axisLeft()
              .scale(offsetYScale)
              .ticks(nticks, "s");

        if (this.xMin != 0) {
            this.xLabel.text(`Key (minus ${this.xMin})`);
        } else {
            this.xLabel.text(`Key`);
        }

        if (this.yMin != 0) {
            this.yLabel.text(`Position (minus ${this.yMin})`);
        } else {
            this.yLabel.text("Position");
        }

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

        const cdf = this.cdfDots
              .selectAll("circle")
              .data(this.cdfData);

        cdf.enter()
            .append("circle")
            .merge(cdf)
            .attr("cx", d => this.xScale(d.x))
            .attr("cy", d => this.yScale(d.y))
            .attr("r", 4);

        cdf.exit().remove();
        
        /*this.cdfPoints
            .style("fill", "none")
            .style("stroke", "black")
            .style("stroke-width", "3px")
            .datum(this.cdfData)
            .attr("d", line);*/

        this.rmiPoints
            .style("fill", "none")
            .style("stroke", "red")
            .style("stroke-opacity", "0.75")
            .style("stroke-width", "3px")
            .datum(this.rmiData)
            .attr("d", line);
    }
}
