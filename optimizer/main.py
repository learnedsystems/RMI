import json
import itertools
import os
import pandas as pd
import subprocess
import hashlib
import numpy as np
from scipy.stats import linregress
from sklearn.neighbors import KNeighborsRegressor
import pystache


RMI_PATH = "../target/release/rmi"
RMI_CACHE_DIR = "/ssd1/ryan/rmi_cache"

# define the model search space
TOP_ONLY_LAYERS = ["radix", "bradix"]
ANYWHERE_LAYERS = ["linear", "cubic"]
SPECIALTY_TOP_LAYERS = ["histogram", "loglinear", "normal", "lognormal"]
BRANCHING_FACTORS = list(int(x) for x in 2**np.arange(7, 22, 1))
ALL_TOP_LAYERS = TOP_ONLY_LAYERS + ANYWHERE_LAYERS


def sha(s):
    return hashlib.sha256(s.encode("UTF-8")).hexdigest()[-20:]

def bvl():
    if not os.path.exists("bvl.csv"):
        print("Recomputing bvl...")
        os.system("g++ -ffast-math -march=native -O3 measure_linvbin.cpp")
        os.system("./a.out > bvl.csv")

    df = pd.read_csv("bvl.csv", header=None, names=["method", "err", "time"])
    df["time"] /= 500000

    x = df[df.method == "linear"]["err"]
    y = df[df.method == "linear"]["time"]

    _lslope, _lintercept, lrval, _, _ = linregress(x, y)
    assert lrval > 0.98

    x = df[df.method == "binary"]["err"]
    y = df[df.method == "binary"]["time"]
    _slope, _intercept, _rval, _, _ = linregress(np.log(x), y)

    bin_pred = KNeighborsRegressor(n_neighbors=2)
    bin_pred.fit(df[df.method == "binary"][["err"]], df[df.method == "binary"]["time"])

    lin_pred = KNeighborsRegressor(n_neighbors=2)
    lin_pred.fit(df[df.method == "linear"][["err"]], df[df.method == "linear"]["time"])

    return (lin_pred, bin_pred)


def namespace(layers, bf):
    return sha(f'{layers} {bf}')

def build_initial_configs():
    # first, build a grid of the most likely configs
    configs = []
    for top in ALL_TOP_LAYERS:
        for bot in ANYWHERE_LAYERS:
            for bf in BRANCHING_FACTORS[::3]:
                layers = f"{top},{bot}"
                configs.append({"layers": layers,
                                "branching factor": bf,
                                "namespace": f"nm{namespace(layers, bf)}"})

    # next, build a few tests to see if a speciality layer would help
    for top in SPECIALTY_TOP_LAYERS:
        if top == "histogram":
            for bot in ANYWHERE_LAYERS:
                for bf in [64, 128, 256]:
                    layers = f"{top},{bot}"
                    configs.append({"layers": layers,
                                    "branching factor": bf,
                                    "namespace": f"nm{namespace(layers, bf)}"})
        else:
            # not a histogram
            for bot in ANYWHERE_LAYERS:
                for bf in BRANCHING_FACTORS[::4]:
                    layers = f"{top},{bot}"
                    configs.append({"layers": layers,
                                    "branching factor": bf,
                                    "namespace": f"nm{namespace(layers, bf)}"})
    return configs

def build_configs_for_layers(candidate_layers, current_results):
    next_configs = []
    for candidate in candidate_layers:
        if candidate.startswith("histogram"):
            for bf in [32, 300, 512]:
                next_configs.append({"layers": candidate,
                                     "branching factor": bf,
                                     "namespace": f"nm{namespace(candidate, bf)}"})

        else:
            already_known = (current_results
                             [current_results.layers == candidate]
                             ["branching factor"]
                             .to_list())
            for bf in sorted(set(BRANCHING_FACTORS) - set(already_known)):
                next_configs.append({"layers": candidate,
                                     "branching factor": bf,
                                     "namespace": f"nm{namespace(candidate, bf)}"})
    return next_configs

def cache_path(data_path, conf):
    data_file = data_path.split("/")[-1]
    binary = "binary" if "binary" in conf and conf["binary"] else "linear"
    path = f'{RMI_CACHE_DIR}/{data_file}/{conf["layers"]}{conf["branching factor"]}{binary}'
    return path


def check_rmi_cache(data_path, conf):
    path = cache_path(data_path, conf)
    if os.path.exists(path):
        with open(f"{path}/results.json", "r") as f:
            results = json.load(f)

        assert results["namespace"] == conf["namespace"]
        os.system(f'cp {path}/*.cpp ./')
        os.system(f'cp {path}/*.h ./')
        return results

    return False

def cache_rmi(data_path, result):
    path = cache_path(data_path, result)
    os.makedirs(path, exist_ok=True)
    
    os.system(f'cp opt/{result["namespace"]}.cpp {path}/')
    os.system(f'cp opt/{result["namespace"]}.h {path}/')
    os.system(f'cp opt/{result["namespace"]}_data.h {path}/')

    with open(f"{path}/results.json", "w") as f:
        json.dump(result, f)

    
def parallel_test_rmis(data_path, configs, threads=6, phase=""):
    if len(configs) < threads:
        threads = len(configs)

    os.system("rm -f *.json *.json_results")
    jobs = [[] for _ in range(threads)]
    procs = []
    data = []

    # check to see if any of these configurations are in the cache
    uncached_configs = []
    for conf in configs:
        if cached := check_rmi_cache(data_path, conf):
            data.append(cached)
        else:
            uncached_configs.append(conf)

    print("Models to compute:", len(uncached_configs), "/", len(configs))
    configs = uncached_configs

    for idx, conf in enumerate(configs):
        jobs[idx % threads].append(conf)

    jobs = [x for x in jobs if x]
        
    for idx, workset in enumerate(jobs):
        fn = f"{phase}{idx}.json"
        with open(fn, "w") as f:
            json.dump({"configs": workset}, f, allow_nan=False)
        cmd = f"{RMI_PATH} {data_path} --param-grid {fn}"
            
        procs.append(subprocess.Popen(cmd, shell=True))

    if jobs:
        print("Spawned", threads, "processes with", [len(x) for x in jobs], "jobs each")
    
    for idx, proc in enumerate(procs):
        if proc.wait() != 0:
            print("Failure in RMI construction at idx", idx)
            assert False

    os.system("sync")
    os.system("rm -rf opt/")
    os.system("mkdir opt")
    os.system("mv nm* opt/")
    for idx, ws in enumerate(jobs):
        fn = f"{phase}{idx}.json_results"
        with open(fn, "r") as f:
            results = json.load(f)
            data.extend(results)

            for result in results:
                cache_rmi(data_path, result)

    return sorted(data, key=lambda x: x["namespace"])


def inference(data_path, df):
    cache_mask = []
    cache_vals = []

    for _, row in df.iterrows():
        loc = f"{cache_path(data_path, row)}/inference.txt"
        if os.path.exists(loc):
            cache_mask.append(True)
            with open(loc, "r") as f:
                cache_vals.append(float(f.read()))
        else:
            cache_mask.append(False)

    not_cached_mask = [not x for x in cache_mask]

    print("Of", len(df), "inference times to compute,", len(cache_vals), "are cached")

    if len(cache_vals) != len(df):
        with open("bench.cpp", "r") as f:
            template = f.read()

        with open("to_build.cpp", "w") as f:
            f.write(pystache.render(template, 
                                    {"filename": data_path, 
                                     "namespaces": df[not_cached_mask].namespace.tolist()}))

        if os.system("make -j 40") != 0:
            print("Error compiling inference program!")

        os.system("./a.out > inference.txt")
        with open("inference.txt") as f:
            times = list(int(x.strip()[:-2]) / 100000.0 for x in f)

        for time, (_, row) in zip(times, df[not_cached_mask].iterrows()):
            loc = f"{cache_path(data_path, row)}/inference.txt"
            with open(loc, "w") as f:
                f.write(str(time))
        
    combined_times = []
    for cached in cache_mask:
        if cached:
            combined_times.append(cache_vals.pop(0))
        else:
            combined_times.append(times.pop(0))
        
    return combined_times

def predict_all(df, lin_pred, bin_pred):
    df["pred_time_lin"] = df.inference + lin_pred.predict(df[["max error"]])
    df["pred_time_bin"] = df.inference + bin_pred.predict(df[["max error"]])
    df["pred_time"] = df[["pred_time_lin", "pred_time_bin"]].min(axis=1)


def pareto_mask(df, props=["size linear search", "max error", "inference"], ignore_star=False):
    # find Pareto efficient RMIs
    mask = []
    for idx1, el1 in df.iterrows():
        my_props = el1[props]
        starred = "star" in el1 and el1["star"]

        if starred and (not ignore_star):
            mask.append(True)
            continue
        
        for idx2, el2 in df.iterrows():
            if idx1 == idx2:
                continue

            other_props = el2[props]
            if (other_props <= my_props).all():
                mask.append(False)
                break
        else:
            mask.append(True)
    return mask


def measure_rmis(data_path, configs):
    os.system("rm -rf SOSD/build")
    os.system("rm -rf SOSD/competitors/rmi/nm*")

    # check to see what times we already have cached
    uncached_configs = []
    cache_mask = []
    cached_values = []
    for config in configs:
        path = cache_path(data_path, config) + "/search.txt"
        if os.path.exists(path):
            cache_mask.append(True)
            with open(path, "r") as f:
                cached_values.append(float(f.read()))
        else:
            cache_mask.append(False)
            uncached_configs.append(config)
    times = []

    if uncached_configs:
        for config in uncached_configs:
            ns = config["namespace"]
            os.system(f"cp opt/{ns}.cpp SOSD/competitors/rmi/")
            os.system(f"cp opt/{ns}_data.h SOSD/competitors/rmi/")
            os.system(f"cp opt/{ns}.h SOSD/competitors/rmi/")

        with open("SOSD/benchmark.cc.mustache", "r") as f:
            template = f.read()

        with open("SOSD/benchmark.cc", "w") as f:
            f.write(pystache.render(template, 
                                    {"candidates": uncached_configs}))

        os.system("cd SOSD && scripts/prepare.sh")
        os.system("cd SOSD && " +
                  f"build/benchmark {data_path} {data_path}_equality_lookups_10M > times.txt")

        with open("SOSD/times.txt", "r") as f:
            for l in f:
                if not l.startswith("RESULT"):
                    continue
                if "," not in l:
                    continue
                result = l.split(",")[1]
                if result == "-1":
                    times.append(float("NaN"))
                else:
                    times.append(float(result))


        for config, time in zip(uncached_configs, times):
            path = cache_path(data_path, config) + "/search.txt"
            with open(path, "w") as f:
                f.write(str(time))

    all_times = []
    for cached in cache_mask:
        if cached:
            all_times.append(cached_values.pop(0))
        else:
            all_times.append(times.pop(0))
            
    return all_times

def optimize(data_path):
    lin_pred, bin_pred = bvl()

    print("Compiling RMI learner...")
    os.system("cd .. && cargo build --release")

    initial_configs = build_initial_configs()
    print("Testing", len(initial_configs), "initial configurations.")
    step1_results = parallel_test_rmis(data_path, initial_configs, phase="step1")
    step1_results = pd.DataFrame(step1_results)

    # measure inference time
    step1_results["inference"] = inference(data_path, step1_results)

    # make predictions
    predict_all(step1_results, lin_pred, bin_pred)

    # always test bradix linear models
    step1_results["star"] = False
    step1_results.loc[step1_results["layers"] == "bradix,linear", "star"] = True

    mask = pareto_mask(step1_results)
    pareto = step1_results[mask]
    print("Of", len(step1_results), "tested models in step 1,", len(pareto), "are on the front")
    print("Layers:", set(pareto["layers"]))
    
    # step 2: expand search on the front
    candidate_layers = set(pareto["layers"])
    step2_configs = build_configs_for_layers(candidate_layers, step1_results)

    print("Testing", len(step2_configs), "additional configurations")
    step2_results = parallel_test_rmis(data_path, step2_configs, phase="step2")
    step2_results = pd.DataFrame(step2_results)

    # measure inference time
    step2_results["inference"] = inference(data_path, step2_results)

    # make predictions
    predict_all(step2_results, lin_pred, bin_pred)
    print("Step 2 ends with", len(step2_results), "models considered")
    
    
    # combine the results together, explicitly setting binary
    all_results = pd.concat((step1_results, step2_results), sort=False).reset_index(drop=True)
    all_results["binary"] = all_results["pred_time_lin"] >= all_results["pred_time_bin"]
    all_results.fillna(False, inplace=True)
    print("Step 2 and step 1 combine to form", len(all_results), "models")

    # build configs for all models on the front
    mask = pareto_mask(all_results)
    pareto_results = all_results[mask]

    
    step3_configs = []
    for _idx, row in pareto_results.iterrows():
        step3_configs.append(row.to_dict())

    step3_results = parallel_test_rmis(data_path, step3_configs, phase="step3")
    step3_times = measure_rmis(data_path, step3_results)
    step3_results = pd.DataFrame(step3_results)
    step3_results["measured"] = step3_times

    # check for any failures
    failed_mask = step3_results["measured"].isna()
    for _, row in step3_results[failed_mask][["layers", "branching factor"]].iterrows():
        print("RMI failure:", row)

    print("At the end of step 3, we have evaluated", len(step3_results), "models")

    # expand on the front
    mask = pareto_mask(step3_results, props=["size linear search", "measured"], ignore_star=True)
    front = step3_results[mask]
    step4_configs = []
    for _idx, row in front.iterrows():
        layers = row["layers"]
        bf = row["branching factor"]
        step4_configs.append({
            "layers": row["layers"],
            "branching factor": int(bf * 1.5),
            "namespace": "nm" + namespace(layers, int(bf * 1.5)),
            "binary": row["binary"]
        })
        step4_configs.append({
            "layers": row["layers"],
            "branching factor": int(bf * 0.5),
            "namespace": "nm" + namespace(layers, int(bf * 0.5)),
            "binary": row["binary"]
        })

    print("Expanding the front, searching", len(step4_configs), "additional models")
    step4_results = parallel_test_rmis(data_path, step4_configs, phase="step4")
    step4_times = measure_rmis(data_path, step4_results)
    step4_results = pd.DataFrame(step4_results)
    step4_results["measured"] = step4_times
    print("Got results for", len(step4_results), "additional models")

    # compute sizes
    results = pd.concat((step3_results, step4_results), sort=False).reset_index(drop=True)
    print("Final number of models evaluated:", len(results))
    mask = pareto_mask(results, props=["size linear search", "measured"], ignore_star=True)
    print("Size of front:", sum(mask))
    results["size"] = (results["binary"].astype(np.int) * results["size binary search"] +
                       ((results["binary"].astype(np.int) + 1) % 2) * results["size linear search"])
    results["front"] = mask
    results[
        ["layers", "branching factor", "size",
         "average error", "max error", "binary", "front", "measured"]
    ].sort_values("measured").to_csv("out.csv", index=False)
    print("Results saved to out.csv")
    os.system("cat out.csv")
    

optimize("/home/ryan/SOSD/data/uniform_sparse_200M_uint64")
