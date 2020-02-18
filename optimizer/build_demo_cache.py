from main import parallel_test_rmis, params_to_rmi_config, measure_rmis, build_shared_object

DS = ["osm_cellids_200M_uint64",
      "books_200M_uint64",
      "fb_200M_uint64",
      "lognormal_200M_uint64",
      "normal_200M_uint64"]

TOP_LAYERS = ["linear", "radix", "bradix", "linear_spline", "cubic", "loglinear", "normal", "lognormal"]
BOTTOM_LAYERS = ["linear", "linear_spline", "cubic", "loglinear"]
BFACTORS = [1024, 4096, 16284, 65536, 262144, 1048576, 2097152]

for ds in DS:
    data_path = "/home/ryan/SOSD/data/" + ds
    all_configs = []
    
    for tl in TOP_LAYERS:
        for bl in BOTTOM_LAYERS:
            for bf in BFACTORS:
                config = params_to_rmi_config(f"{tl},{bl}", bf)
                config["binary"] = False
                all_configs.append(config)

    parallel_test_rmis(data_path, all_configs)

    all_configs = []
    for tl in TOP_LAYERS:
        for bl in BOTTOM_LAYERS:
            for bf in BFACTORS:
                config = params_to_rmi_config(f"{tl},{bl}", bf)
                config["binary"] = True
                all_configs.append(config)

    measure_rmis(data_path, all_configs)

