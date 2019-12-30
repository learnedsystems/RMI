import numpy as np
from flask import Flask, send_from_directory
import os
from ctypes import cdll, c_ulonglong
from main import parallel_test_rmis, params_to_rmi_config
from functools import lru_cache


MAX_RESOLUTION = 1000

DLL_COUNT = 0

def get_range(data_file, start, stop):
    arr = np.memmap(data_file, dtype=np.uint64, mode="r")[1:]
    idxes = np.floor(np.linspace(start, stop, MAX_RESOLUTION, endpoint=False)).astype(np.uint64)
    keys = arr[idxes]
    
    return (keys, idxes)


@lru_cache
def load_rmi(data_path, layers, bf):
    global DLL_COUNT
    config = params_to_rmi_config(layers, bf)
    parallel_test_rmis(data_path, [config])
    os.system("make clean")
    os.system("make -j objects")
    # need to generate unique names because Linux caches
    # dylibs by file path XD
    new_name = f'opt/l{DLL_COUNT}{config["namespace"]}.so'
    DLL_COUNT += 1
    os.system(f'cp opt/{config["namespace"]}.so {new_name}')
    os.system("sync")
    rmi = cdll.LoadLibrary(new_name)
    conv = lambda x: rmi.lookup(c_ulonglong(int(x)))
    return np.vectorize(conv)
    

def get_range_rmi(data_path, layers, bf, start, stop):
    lookup = load_rmi(data_path, layers, bf)
    keys = get_range(data_path, start, stop)[0]
    pred_idxes = lookup(keys)

    return (keys, pred_idxes)

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
    print(fn)
    x, y = get_range_rmi("/home/ryan/SOSD/data/" + fn, layers, bf, start, stop)
    
    data = []
    for xv, yv in zip(x, y):
        data.append({"x": int(xv), "y": int(yv)})
    return {"data": data}


@app.route('/<path:path>')
def send_static(path):
    return send_from_directory("static", path)
