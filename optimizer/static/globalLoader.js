// https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Using_promises
const wait = ms => new Promise(resolve => setTimeout(resolve, ms));
let indicatorStatus = false;

function showProgressIndicator() {
    indicatorStatus = true;
    wait(500).then(() => {
        if (indicatorStatus) {
            console.log("showing!");
            d3.select("#global-loader").style("display", "block");
            d3.select("#content").style("display", "none");
        }
    });
}

function hideProgressIndicator() {
    console.log("hiding");
    indicatorStatus = false;
    d3.select("#global-loader").style("display", "none");
    d3.select("#content").style("display", "block");
}
