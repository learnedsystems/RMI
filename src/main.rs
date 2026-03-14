// < begin copyright >
// Copyright Ryan Marcus 2020
//
// See root directory of this project for license terms.
//
// < end copyright >

#![allow(clippy::needless_return)]

#[macro_use]
mod load;

use load::{load_data, DataType};
use rmi_lib::optimizer;
use rmi_lib::KeyType;
use rmi_lib::RMIModel;
use rmi_lib::{train, train_bounded};

use json::*;
use log::*;
use rayon::prelude::*;
use std::f64;
use std::fs;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use clap::{App, Arg};
use indicatif::{ProgressBar, ProgressStyle};

fn attach_optimal_pla_epsilon(models: &str, epsilon: Option<usize>) -> String {
    let epsilon = match epsilon {
        Some(v) => v,
        None => return models.to_string(),
    };

    models
        .split(',')
        .map(|layer| {
            if layer == "optimal_pla" {
                format!("optimal_pla:{}", epsilon)
            } else {
                layer.to_string()
            }
        })
        .collect::<Vec<_>>()
        .join(",")
}

fn main() {
    env_logger::init();

    let matches = App::new("RMI Learner")
        .version("0.1")
        .author("Ryan Marcus <ryan@ryanmarc.us>")
        .about("Learns recursive model indexes")
        .arg(Arg::with_name("input")
             .help("Path to input file containing data")
             .index(1).required(true))
        .arg(Arg::with_name("namespace")
             .help("Namespace to use in generated code")
             .index(2).required(false))
        .arg(Arg::with_name("models")
             .help("Comma-separated list of model layers, e.g. linear,linear")
             .index(3).required(false))
        .arg(Arg::with_name("branching factor")
             .help("Branching factor between each model level")
             .index(4).required(false))
        .arg(Arg::with_name("no-code")
             .long("no-code")
             .help("Skip code generation"))
        .arg(Arg::with_name("dump-ll-model-data")
             .long("dump-ll-model-data")
             .value_name("model_index")
             .help("dump the data used to train the last-level model at index"))
        .arg(Arg::with_name("dump-ll-errors")
             .long("dump-ll-errors")
             .help("dump the errors of each last-level model to ll_errors.json"))
        .arg(Arg::with_name("stats-file")
             .long("stats-file")
             .short("s")
             .value_name("file")
             .help("dump statistics about the learned model into the specified file"))
        .arg(Arg::with_name("param-grid")
             .long("param-grid")
             .value_name("file")
             .help("train the RMIs specified in the JSON file and report their errors"))
        .arg(Arg::with_name("data-path")
             .long("data-path")
             .short("d")
             .value_name("dir")
             .help("exports parameters to files in this directory (default: rmi_data)"))
        .arg(Arg::with_name("output-path")
             .long("output-path")
             .short("o")
             .value_name("dir")
             .help("exports generated model files to this directory (default: current directory)"))
        .arg(Arg::with_name("binary-output")
             .long("binary-output")
             .value_name("PATH")
             .help("Path to write binary model file"))
        .arg(Arg::with_name("no-errors")
             .long("no-errors")
             .help("do not save last-level errors, and modify the RMI function signature"))
        .arg(Arg::with_name("threads")
             .long("threads")
             .short("t")
             .value_name("count")
             .help("number of threads to use for optimization, default = 4"))
        .arg(Arg::with_name("bounded")
             .long("bounded")
             .value_name("line_size")
             .help("construct an error-bounded RMI using the cachefix method for the given line size"))
        .arg(Arg::with_name("max-size")
             .long("max-size")
             .value_name("BYTES")
             .help("uses the optimizer fo find an RMI with a size less than specified"))
        .arg(Arg::with_name("disable-parallel-training")
             .long("disable-parallel-training")
             .help("disables training multiple RMIs in parallel"))
        .arg(Arg::with_name("zero-build-time")
             .long("zero-build-time")
             .help("zero out the model build time field"))
        .arg(Arg::with_name("optimize")
             .long("optimize")
             .value_name("file")
             .help("Search for Pareto efficient RMI configurations. Specify the name of the output file."))
        .arg(Arg::with_name("epsilon")
             .long("epsilon")
             .value_name("N")
             .help("Epsilon for optimal_pla layers. Applied when a layer is specified as plain optimal_pla."))
        .get_matches();

    // set the max number of threads to 4 by default, otherwise Rayon goes
    // crazy on larger machines and allocates too many workers for folds / reduces
    let num_threads = matches
        .value_of("threads")
        .map(|x| x.parse::<usize>().unwrap())
        .unwrap_or(4);
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_threads)
        .build_global()
        .unwrap();

    let fp = matches.value_of("input").unwrap();

    let data_dir = matches.value_of("data-path").unwrap_or("rmi_data");
    let output_dir = matches.value_of("output-path").unwrap_or(".");
    let binary_output = matches.value_of("binary-output").map(|s| s.to_string());
    let pla_epsilon = matches.value_of("epsilon").map(|v| {
        v.parse::<usize>()
            .expect("--epsilon must be a non-negative integer")
    });

    if matches.value_of("namespace").is_some() && matches.value_of("param-grid").is_some() {
        panic!("Can only specify one of namespace or param-grid");
    }

    info!("Reading {}...", fp);

    let mut key_type = KeyType::U64;
    let (num_rows, data) = if fp.contains("uint64") {
        load_data(&fp, DataType::UINT64)
    } else if fp.contains("uint32") {
        load_data(&fp, DataType::UINT32)
    } else if fp.contains("f64") {
        key_type = KeyType::F64;
        load_data(&fp, DataType::FLOAT64)
    } else {
        panic!("Data file must contain uint64, uint32, or f64.");
    };

    if matches.is_present("optimize") {
        let results = dynamic!(optimizer::find_pareto_efficient_configs, data, 10);

        optimizer::RMIStatistics::display_table(&results);

        let nmspc_prefix = if matches.value_of("namespace").is_some() {
            matches.value_of("namespace").unwrap()
        } else {
            let path = Path::new(fp);
            path.file_name()
                .map(|s| s.to_str())
                .unwrap_or(Some("rmi"))
                .unwrap()
        };

        let grid_specs: Vec<JsonValue> = results
            .into_iter()
            .enumerate()
            .map(|(idx, v)| {
                let nmspc = format!("{}_{}", nmspc_prefix, idx);
                v.to_grid_spec(&nmspc)
            })
            .collect();

        let grid_specs_json = object!("configs" => grid_specs);
        let fp = matches.value_of("optimize").unwrap();
        let f = File::create(fp).expect("Could not write optimization results file");
        let mut bw = BufWriter::new(f);
        grid_specs_json.write(&mut bw).unwrap();
        return;
    }

    // if we aren't optimizing, we should make sure the RMI data directory exists.
    if !Path::new(data_dir).exists() {
        info!(
            "The RMI data directory specified {} does not exist. Creating it.",
            data_dir
        );
        std::fs::create_dir_all(data_dir)
            .expect("The RMI data directory did not exist, and it could not be created.");
    }

    if !Path::new(output_dir).exists() {
        info!(
            "The RMI output directory specified {} does not exist. Creating it.",
            output_dir
        );
        std::fs::create_dir_all(output_dir)
            .expect("The RMI output directory did not exist, and it could not be created.");
    }

    if let Some(param_grid) = matches.value_of("param-grid").map(|x| x.to_string()) {
        if binary_output.is_some() {
            warn!("Binary output is ignored when training multiple RMIs from a parameter grid");
        }

        let pg = {
            let raw_json = fs::read_to_string(param_grid.clone()).unwrap();
            let mut as_json = json::parse(raw_json.as_str()).unwrap();
            as_json["configs"].take()
        };

        let mut to_test = Vec::new();
        if let JsonValue::Array(v) = pg {
            for el in v {
                let layers =
                    attach_optimal_pla_epsilon(el["layers"].as_str().unwrap(), pla_epsilon);
                let branching = el["branching factor"].as_u64().unwrap();
                let namespace = match el["namespace"].as_str() {
                    Some(s) => Some(String::from(s)),
                    None => None,
                };

                to_test.push((layers, branching, namespace));
            }

            trace!("# RMIs to train: {}", to_test.len());

            let pbar = ProgressBar::new(to_test.len() as u64);
            pbar.set_style(
                ProgressStyle::default_bar().template("{pos} / {len} ({msg}) {wide_bar} {eta}"),
            );

            let train_func =
                |(models, branch_factor, namespace): &(String, u64, Option<String>)| {
                    trace!(
                        "Training RMI {} with branching factor {}",
                        models,
                        *branch_factor
                    );

                    let loc_data = data.soft_copy();
                    let mut trained_model = dynamic!(train, loc_data, models, *branch_factor);

                    let size_bs = rmi_lib::rmi_size(&trained_model);

                    let result_obj = object! {
                        "layers" => models.clone(),
                        "branching factor" => *branch_factor,
                        "average error" => trained_model.model_avg_error as f64,
                        "average error %" => trained_model.model_max_error as f64
                            / num_rows as f64 * 100.0,
                        "average l2 error" => trained_model.model_avg_l2_error as f64,
                        "average log2 error" => trained_model.model_avg_log2_error,
                        "max error" => trained_model.model_max_error,
                        "max error %" => trained_model.model_max_error as f64
                            / num_rows as f64 * 100.0,
                        "max log2 error" => trained_model.model_max_log2_error,
                        "size binary search" => size_bs,
                        "namespace" => namespace.clone()
                    };

                    if matches.is_present("zero-build-time") {
                        trained_model.build_time = 0;
                    }

                    if let Some(nmspc) = namespace {
                        rmi_lib::output_rmi(
                            &nmspc,
                            trained_model,
                            data_dir,
                            output_dir,
                            key_type,
                            true,
                        )
                        .unwrap();
                    }

                    pbar.inc(1);
                    return result_obj;
                };

            let results: Vec<JsonValue> = if matches.is_present("disable-parallel-training") {
                trace!("Training models sequentially");
                to_test.iter().map(train_func).collect()
            } else {
                trace!("Training models in parallel");
                to_test.par_iter().map(train_func).collect()
            };

            //let results: Vec<JsonValue> = to_test
            //.par_iter().map(
            pbar.finish();

            let f = File::create(format!("{}_results", param_grid))
                .expect("Could not write results file");
            let mut bw = BufWriter::new(f);
            let json_results = object! { "results" => results };
            json_results.write(&mut bw).unwrap();
        } else {
            panic!("Configs must have an array as its value");
        }
    } else if matches.value_of("namespace").is_some() {
        let namespace = matches.value_of("namespace").unwrap().to_string();
        let mut trained_model = match matches.value_of("max-size") {
            None => {
                // assume they gave a model spec
                let models =
                    attach_optimal_pla_epsilon(matches.value_of("models").unwrap(), pla_epsilon);
                let branch_factor = matches
                    .value_of("branching factor")
                    .unwrap()
                    .parse::<u64>()
                    .unwrap();

                let trained_model = match matches.value_of("bounded") {
                    None => dynamic!(train, data, &models, branch_factor),
                    Some(s) => {
                        let line_size = s
                            .parse::<usize>()
                            .expect("Line size must be a positive integer.");
                        let d_u64 = data
                            .into_u64()
                            .expect("Can only construct a bounded RMI on u64 data.");
                        train_bounded(&d_u64, &models, branch_factor, line_size)
                    }
                };
                trained_model
            }
            Some(max_size_str) => {
                let max_size = max_size_str.parse::<usize>().unwrap();
                info!("Constructing RMI with size less than {}", max_size);

                let trained_model = dynamic!(rmi_lib::train_for_size, data, max_size);
                trained_model
            }
        };

        let no_errors = matches.is_present("no-errors");
        info!(
            "Model build time: {} ms",
            trained_model.build_time / 1_000_000
        );

        info!(
            "Average model error: {} ({}%)",
            trained_model.model_avg_error as f64,
            trained_model.model_avg_error / num_rows as f64 * 100.0
        );
        info!(
            "Average model L2 error: {}",
            trained_model.model_avg_l2_error
        );
        info!(
            "Average model log2 error: {}",
            trained_model.model_avg_log2_error
        );
        info!(
            "Max model log2 error: {}",
            trained_model.model_max_log2_error
        );
        info!(
            "Max model error on model {}: {} ({}%)",
            trained_model.model_max_error_idx,
            trained_model.model_max_error,
            trained_model.model_max_error as f64 / num_rows as f64 * 100.0
        );

        if let Some(path) = binary_output.as_ref() {
            let rmi_model = RMIModel::from_trained(&trained_model, key_type, !no_errors);
            rmi_model
                .save_binary(path)
                .expect("Failed to save binary RMI model");
            info!("Saved binary RMI model to {}", path);
        }

        if !matches.is_present("no-code") {
            if matches.is_present("zero-build-time") {
                trained_model.build_time = 0;
            }

            rmi_lib::output_rmi(
                &namespace,
                trained_model,
                data_dir,
                output_dir,
                key_type,
                !no_errors,
            )
            .unwrap();
        } else {
            trace!("Skipping code generation due to CLI flag");
        }
    } else {
        trace!("Must specify either a name space or a parameter grid.");
    }
}
