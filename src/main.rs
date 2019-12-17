// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
#![allow(clippy::needless_return)]

mod codegen;
mod load;
mod models;
mod train;

use load::{load_data, DataType};
use train::train;

use json::*;
use log::*;
use std::f64;
use std::fs::File;
use std::io::BufWriter;
use std::io::Write;
use std::time::SystemTime;
use std::fs;

use indicatif::{ProgressBar, ProgressStyle};
use clap::{App, Arg};

fn main() {
    let start_time = SystemTime::now();
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
        .arg(Arg::with_name("downsample")
             .long("downsample")
             .value_name("factor")
             .help("downsample the data by an integer factor (for faster training)"))
        .arg(Arg::with_name("last-layer-errors")
             .long("last-layer-errors")
             .short("e")
             .help("cause the lookup function to return a maximum search distance in addition to a guess"))
        .arg(Arg::with_name("dump-ll-model-data")
             .long("dump-ll-model-data")
             .value_name("model_index")
             .help("dump the data used to train the last-level model at index"))
        .arg(Arg::with_name("stats-file")
             .long("stats-file")
             .short("s")
             .value_name("file")
             .help("dump statistics about the learned model into the specified file"))
        .arg(Arg::with_name("param-grid")
             .long("param-grid")
             .value_name("file")
             .help("train the RMIs specified in the JSON file and report their errors"))
        .get_matches();

    let fp = matches.value_of("input").unwrap();
    let downsample = matches
        .value_of("downsample")
        .map(|x| x.parse::<usize>().unwrap())
        .unwrap_or(1);
    
    if matches.value_of("namespace").is_some() && matches.value_of("param-grid").is_some() {
        panic!("Can only specify one of namespace or param-grid");
    }
    
    info!("Reading {}...", fp);

    let (num_rows, data) = if fp.contains("uint64") {
        load_data(&fp, DataType::UINT64, downsample)
    } else {
        load_data(&fp, DataType::UINT32, downsample)
    };
    
    let load_time = SystemTime::now()
        .duration_since(start_time)
        .map(|d| d.as_nanos())
        .unwrap_or(std::u128::MAX);
    info!("Data read time: {} ms", load_time / 1_000_000);



    if let Some(param_grid) = matches.value_of("param-grid").map(|x| x.to_string()) {
        let pg = {
            let raw_json = fs::read_to_string(param_grid.clone()).unwrap();
            let mut as_json = json::parse(raw_json.as_str()).unwrap();
            let configs = as_json["configs"].take();
            configs
        };

        let mut to_test = Vec::new();
        if let JsonValue::Array(v) = pg {
            for el in v {
                let layers = String::from(el["layers"].as_str().unwrap());
                let branching = el["branching factor"].as_u64().unwrap();
                let namespace = match el["namespace"].as_str() {
                    Some(s) => Some(String::from(s)),
                    None => None
                };
                
                to_test.push((layers, branching, namespace));
            }

            trace!("# RMIs to train: {}", to_test.len());

            let bar = ProgressBar::new(to_test.len() as u64);
            bar.set_style(ProgressStyle::default_bar()
                          .template("{pos} / {len} ({msg}) {wide_bar} {eta}"));
            
            let mut results = json::JsonValue::new_array();
            
            for (models, branch_factor, namespace) in to_test {
                trace!("Training RMI {} with branching factor {}", models, branch_factor);
                bar.set_message(format!("{} {}", models, branch_factor).as_str());
                // TODO this copy should not be required
                let trained_model = train(data.clone(), &models, branch_factor);
                
                let size_bs = codegen::rmi_size(&trained_model.rmi, true);
                let size_ls = codegen::rmi_size(&trained_model.rmi, false);
                results.push(object! {
                    "layers" => models.clone(),
                    "branching factor" => branch_factor,
                    "average error" => trained_model.model_avg_error as f64,
                    "average error %" => trained_model.model_max_error as f64 / num_rows as f64 * 100.0,
                    "max error" => trained_model.model_max_error,
                    "max error %" => trained_model.model_max_error as f64 / num_rows as f64 * 100.0,
                    "size binary search" => size_bs,
                    "size linear search" => size_ls,
                    "namespace" => namespace.clone()
                }).unwrap();

                if let Some(nmspc) = namespace {
                    codegen::output_rmi(
                        &nmspc,
                        false,
                        trained_model,
                        num_rows,
                        0).unwrap();

                }
                
                bar.inc(1);
            }
            bar.finish();

            let f = File::create(format!("{}_results", param_grid)).expect("Could not write results file");
            let mut bw = BufWriter::new(f);
            results.write(&mut bw).unwrap();
            
        } else {
            panic!("Configs must have an array as its value");
        }

    } else if matches.value_of("namespace").is_some() {
        let namespace = matches.value_of("namespace").unwrap().to_string();
        let models = matches.value_of("models").unwrap();
        
        let branch_factor = matches
            .value_of("branching factor")
            .unwrap()
            .parse::<u64>()
            .unwrap();
        
       
        
        let last_layer_errors = matches.is_present("last-layer-errors");
        let trained_model = train(data, models, branch_factor);
        
        let build_time = SystemTime::now()
            .duration_since(start_time)
            .map(|d| d.as_nanos())
            .unwrap_or(std::u128::MAX);
        info!("Model build time: {} ms", build_time / 1_000_000);

        info!(
            "Average model error: {} ({}%)",
            trained_model.model_avg_error as f64,
            trained_model.model_avg_error / num_rows as f64 * 100.0
        );
        info!(
            "Max model error on model {}: {} ({}%)",
            trained_model.model_max_error_idx,
            trained_model.model_max_error,
            trained_model.model_max_error as f64 / num_rows as f64 * 100.0
        );
        
        match matches.value_of("stats-file") {
            None => {}
            Some(stats_fp) => {
                let output = object! {
                    "average error" => trained_model.model_avg_error,
                    "max error" => trained_model.model_max_error
                };
                
                let f = File::create(stats_fp.to_string()).expect("Could not write stats file");
                let mut bw = BufWriter::new(f);
                writeln!(bw, "{}", output.dump()).unwrap();
            }
        }
        
        if !matches.is_present("no-code") {
            codegen::output_rmi(
                &namespace,
                last_layer_errors,
                trained_model,
                num_rows,
                build_time).unwrap();
        } else {
            trace!("Skipping code generation due to CLI flag");
        }
    } else {
        trace!("Must specify either a name space or a parameter grid.");
    }
}
