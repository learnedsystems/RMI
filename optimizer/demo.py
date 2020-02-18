import numpy as np
from flask import Flask, send_from_directory
import os
from ctypes import cdll, c_ulonglong
from main import parallel_test_rmis, params_to_rmi_config, measure_rmis, build_shared_object
from functools import lru_cache
import time

MAX_RESOLUTION = 1000



def get_range(data_file, start, stop, res=MAX_RESOLUTION):
    arr = np.memmap(data_file, dtype=np.uint64, mode="r")[1:]
    idxes = np.floor(np.linspace(start, stop, res, endpoint=False)).astype(np.uint64)
    keys = arr[idxes]
    
    return (keys, idxes)


@lru_cache
def load_rmi(data_path, layers, bf):
    config = params_to_rmi_config(layers, bf)
    path = build_shared_object(data_path, config)
    rmi = cdll.LoadLibrary(path)
    conv = lambda x: rmi.lookup(c_ulonglong(int(x)))
    return np.vectorize(conv)

@lru_cache
def rmi_stats(data_path, layers, bf):
    config = params_to_rmi_config(layers, bf)
    results = parallel_test_rmis(data_path, [config])[0]
    return results

def get_range_rmi(data_path, layers, bf, start, stop, res=MAX_RESOLUTION):
    lookup = load_rmi(data_path, layers, bf)
    keys = get_range(data_path, start, stop, res=res)[0]
    pred_idxes = lookup(keys)

    return (keys, pred_idxes)

def get_rmi_variance(data_path, layers, bf):
    true_vals = np.array(get_range(data_path, 0, 200_000_000, res=1_000_000)[1])
    pred_vals = np.array(get_range_rmi(data_path, layers, bf, 0, 200_000_000, res=1_000_000)[1])

    errs = np.sort(np.abs(true_vals - pred_vals))
    sample_idxes = np.linspace(0, len(errs), num=MAX_RESOLUTION, endpoint=False).astype(np.int32)
    perc = errs[sample_idxes].tolist()
    perc.append(errs[-1])

    return perc[1:]

app = Flask(__name__)

DATASETS = set([
    "lognormal_200M_uint64",
    "osm_cellids_200M_uint64",
    "books_200M_uint64",
    "fb_200M_uint64",
    "normal_200M_uint64"
])

@app.route('/data/<fn>/<int:start>/<int:stop>')
def serve_data(fn, start, stop):
    assert fn in DATASETS
    x, y = get_range("/home/ryan/SOSD/data/" + fn, start, stop)

    data = []
    for xv, yv in zip(x, y):
        data.append({"x": int(xv), "y": int(yv)})
    return {"data": data}
    

@app.route('/rmi/<fn>/<layers>/<int:bf>/<int:start>/<int:stop>')
def serve_rmi(fn, layers, bf, start, stop):
    assert fn in DATASETS
    x, y = get_range_rmi("/home/ryan/SOSD/data/" + fn, layers, bf, start, stop)
    
    data = []
    for xv, yv in zip(x, y):
        data.append({"x": int(xv), "y": int(yv)})
    return {"data": data,
            "stats": rmi_stats(fn, layers, bf)}

@app.route('/variance/<fn>/<layers>/<int:bf>')
def serve_variance(fn, layers, bf):
    assert fn in DATASETS
    percs = get_rmi_variance("/home/ryan/SOSD/data/" + fn, layers, bf)
    return {"results": percs}

@app.route("/measure/<fn>/<layers>/<int:bf>")
def measure(fn, layers, bf):
    config = params_to_rmi_config(layers, bf)
    config["binary"] = True
    results = measure_rmis("/home/ryan/SOSD/data/" + fn, [config])
    time.sleep(2.5)
    return {"result": results}


@app.route('/<path:path>')
def send_static(path):
    return send_from_directory("static", path)
