// https://stackoverflow.com/questions/10420352/converting-file-size-in-bytes-to-human-readable-string/10420404
function humanFileSize(size) {
    if (size < 1024) return size + ' B';
    let i = Math.floor(Math.log(size) / Math.log(1024));
    let num = (size / Math.pow(1024, i));
    let round = Math.round(num);
    num = round < 10 ? num.toFixed(2) : round < 100 ? num.toFixed(1) : round;
    return `${num} ${'KMGTPEZY'[i-1]}B`;
}


function makePlotForStep(step) {
    Plotly.d3.csv(`/osm_cellids_200M_uint64/${step}_out.csv`, function(err, rows){
        function unpack(rows, key, front, div) {
	    return rows.filter(row => row.front == front).map(row => row[key] / div);
        }

        function unpackText(rows, key, front) {
	    return rows.filter(row => row.front == front).map(row => row[key]);
        }

        function buildTooltips(rows, front) {
            return rows.filter(row => row.front == front)
                .map(row => {
                    return `
<b>${row["layers"]} ${row["branching factor"]}</b><br>
Inference: ${parseFloat(row["inference"]).toFixed(2)} ns<br>
Error: ${row["max error"]}<br>
Size: ${humanFileSize(row["size binary search"])}
`;
                });
        }


        var trace1 = {
	    x: unpack(rows, 'max error', "True", 1),
            y: unpack(rows, 'inference', "True", 1),
            z: unpack(rows, 'size binary search', "True", 1024*1024),
            text: buildTooltips(rows, "True"),
            hoverinfo: "text",
	    mode: 'markers',
	    marker: {
                color: "green",
	        size: 3,
	        line: {
		    color: 'black',
		    width: 1},
	        opacity: 1},
            name: "On front",
	    type: 'scatter3d'
        };

        var trace2 = {
	    x: unpack(rows, 'max error', "False", 1),
            y: unpack(rows, 'inference', "False", 1),
            z: unpack(rows, 'size binary search', "False", 1024*1024),
            text: buildTooltips(rows, "False"),
            hoverinfo: "text",
	    mode: 'markers',
	    marker: {
                color: "lightgray",
	        size: 3,
	        line: {
		    color: 'black',
		    width: 1},
	        opacity: 1.0},
            name: "Off front",
	    type: 'scatter3d'
        };

        var data = [trace1, trace2];
        var layout = {
            margin: {
	        l: 0,
	        r: 0,
	        b: 0,
	        t: 0,
            },
            scene: {
                xaxis: { title: "Error", type: "log" },
                yaxis: { title: "Inference (ns)" },
                zaxis: { title: "Size (MB)", type: "log" }
            },
            showlegend: true,
            legend: { x: 1, y: 0.5 }
        };
        Plotly.newPlot(`${step}-plot`, data, layout);
    });
}

makePlotForStep("step1");
makePlotForStep("step2");

function make2DPlotForStep(step, szKey) {
    Plotly.d3.csv(`/osm_cellids_200M_uint64/${step}_out.csv`, function(err, rows){
        function unpack(rows, key, front, div) {
	    return rows.filter(row => row.front == front).map(row => row[key] / div);
        }

        function unpackText(rows, key, front) {
	    return rows.filter(row => row.front == front).map(row => row[key]);
        }

        function buildTooltips(rows, front) {
            return rows.filter(row => row.front == front)
                .map(row => {
                    return `
<b>${row["layers"]} ${row["branching factor"]}</b><br>
Lookup latency: ${parseFloat(row["measured"]).toFixed(2)} ns<br>
Size: ${humanFileSize(row[szKey])}
`;
                });
        }



        var trace1 = {
	    x: unpack(rows, szKey, "True", 1024*1024),
            y: unpack(rows, 'measured', "True", 1),
            text: buildTooltips(rows, "True"),
            hoverinfo: "text",
	    mode: 'markers',
	    marker: {
                color: "green",
	        size: 8,
	        line: {
		    color: 'black',
		    width: 1},
	        opacity: 1},
            name: "On front",
	    type: 'scatter'
        };

        var trace2 = {
	    x: unpack(rows, szKey, "False", 1024*1024),
            y: unpack(rows, 'measured', "False", 1),
            text: buildTooltips(rows, "False"),
            hoverinfo: "text",
	    mode: 'markers',
	    marker: {
                color: "lightgray",
	        size: 8,
	        line: {
		    color: 'black',
		    width: 1},
	        opacity: 1.0},
            name: "Off front",
	    type: 'scatter'
        };

        var data = [trace1, trace2];
        var layout = {
/*            margin: {
	        l: 0,
	        r: 0,
	        b: 0,
	        t: 0,
            },*/

            xaxis: { title: "Size (MB)", type: "log" },
            yaxis: { title: "Latency (ns)" },
            showlegend: true,
            legend: { x: 1, y: 0.5 }
        };
        Plotly.newPlot(`${step}-plot`, data, layout);
    });
}

make2DPlotForStep("step3", "size binary search");
make2DPlotForStep("step4", "size");

d3.csv('/osm_cellids_200M_uint64/step1_out.csv')
    .then(function(data) {
        d3.select("#step1-tested").text(data.length);
        d3.select("#step1-onfront").text(data.filter(d => d["front"] == "True").length);
        d3.select("#step1-offfront").text(data.filter(d => d["front"] == "False").length);

        const modelsSet = new Set(data.filter(d => d["front"] == "True").map(d => d["layers"]));
        let models = [...modelsSet].sort();
        models = models.map(d => `<code>[${d}]</code>`);
        d3.select("#step1-models").html(models.join(", "));

        const elimModelsSet = new Set(data.filter(d => d["front"] == "False").map(d => d["layers"]));
        for (let onfront of modelsSet) {
            elimModelsSet.delete(onfront);
        }
        let elimModels = [...elimModelsSet].sort();
        elimModels = elimModels.map(d => `<code>[${d}]</code>`);
        d3.select("#step1-elimmodels").html(elimModels.join(", "));
        
        data = data.sort((a, b) => {
            const aFront = a["front"] == "True";
            const bFront = b["front"] == "True";

            if (aFront && !bFront) return -1;
            if (!aFront && bFront) return 1;

            return a["size binary search"] - b["size binary search"];
        });
        
        const tbody = d3.select("#tbody-step1");
        // TODO use a d3 group selector
        for (let d of data) {
            const row = tbody.append("tr");
            row.attr("class", d["front"] == "True" ? "table-success" : "table-secondary");
            
            row.append("td").text(d["layers"]);
            row.append("td").text(d["branching factor"]);
            row.append("td").text(d["size binary search"]);
            row.append("td").text(d["inference"]);
            row.append("td").text(d["max error"]);
        }

    });


d3.csv('/osm_cellids_200M_uint64/step2_out.csv')
    .then(function(data) {
        d3.select("#step2-tested").text(data.length);
        d3.select("#step2-onfront").text(data.filter(d => d["front"] == "True").length);
        d3.select("#step2-offfront").text(data.filter(d => d["front"] == "False").length);

        data = data.sort((a, b) => {
            const aFront = a["front"] == "True";
            const bFront = b["front"] == "True";

            if (aFront && !bFront) return -1;
            if (!aFront && bFront) return 1;

            return a["size binary search"] - b["size binary search"];
        });
        
        const tbody = d3.select("#tbody-step2");
        // TODO use a d3 group selector
        for (let d of data) {
            const row = tbody.append("tr");
            row.attr("class", d["front"] == "True" ? "table-success" : "table-secondary");
            
            row.append("td").text(d["layers"]);
            row.append("td").text(d["branching factor"]);
            row.append("td").text(d["size binary search"]);
            row.append("td").text(d["inference"]);
            row.append("td").text(d["max error"]);
        }

    });


d3.csv('/osm_cellids_200M_uint64/step3_out.csv')
    .then(function(data) {
        d3.select("#step3-tested").text(data.length);
        d3.select("#step3-onfront").text(data.filter(d => d["front"] == "True").length);
        d3.select("#step3-offfront").text(data.filter(d => d["front"] == "False").length);

        data = data.sort((a, b) => {
            const aFront = a["front"] == "True";
            const bFront = b["front"] == "True";

            if (aFront && !bFront) return -1;
            if (!aFront && bFront) return 1;

            return a["size binary search"] - b["size binary search"];
        });
        
        const tbody = d3.select("#tbody-step3");
        // TODO use a d3 group selector
        for (let d of data) {
            const row = tbody.append("tr");
            row.attr("class", d["front"] == "True" ? "table-success" : "table-secondary");
            
            row.append("td").text(d["layers"]);
            row.append("td").text(d["branching factor"]);
            row.append("td").text(d["size binary search"]);
            row.append("td").text(d["measured"]);
        }

    });




d3.csv('/osm_cellids_200M_uint64/step4_out.csv')
    .then(function(data) {
        d3.select("#step4-tested").text(data.length);
        d3.select("#step4-onfront").text(data.filter(d => d["front"] == "True").length);
        d3.select("#step4-offfront").text(data.filter(d => d["front"] == "False").length);

        data = data.sort((a, b) => {
            const aFront = a["front"] == "True";
            const bFront = b["front"] == "True";

            if (aFront && !bFront) return -1;
            if (!aFront && bFront) return 1;

            return a["size"] - b["size"];
        });
        
        const tbody = d3.select("#tbody-step4");
        // TODO use a d3 group selector
        for (let d of data) {
            const row = tbody.append("tr");
            row.attr("class", d["front"] == "True" ? "table-success" : "table-secondary");
            
            row.append("td").text(d["layers"]);
            row.append("td").text(d["branching factor"]);
            row.append("td").text(d["size"]);
            row.append("td").text(d["measured"]);
        }

    });
