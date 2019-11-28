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
             .index(2).required(true))
        .arg(Arg::with_name("models")
             .help("Comma-separated list of model layers, e.g. linear,linear")
             .index(3).required(true))
        .arg(Arg::with_name("branching factor")
             .help("Branching factor between each model level")
             .index(4).required(true))
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
        .get_matches();

    let fp = matches.value_of("input").unwrap();
    let namespace = matches.value_of("namespace").unwrap().to_string();
    let models = matches.value_of("models").unwrap();
    let branch_factor = matches
        .value_of("branching factor")
        .unwrap()
        .parse::<u64>()
        .unwrap();
    let downsample = matches
        .value_of("downsample")
        .map(|x| x.parse::<usize>().unwrap())
        .unwrap_or(1);
    let last_layer_errors = matches.is_present("last-layer-errors");

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
        let f1 = File::create(format!("{}.cpp", namespace)).expect("Could not write RMI CPP file");
        let mut bw1 = BufWriter::new(f1);

        let f2 =
            File::create(format!("{}_data.h", namespace)).expect("Could not write RMI data file");
        let mut bw2 = BufWriter::new(f2);

        let f3 = File::create(format!("{}.h", namespace)).expect("Could not write RMI header file");
        let mut bw3 = BufWriter::new(f3);

        let lle = if last_layer_errors {
            Some(trained_model.last_layer_max_l1s)
        } else {
            None
        };

        codegen::generate_code(
            &mut bw1,
            &mut bw2,
            &mut bw3,
            namespace,
            num_rows,
            trained_model.rmi,
            lle,
            build_time,
        )
        .unwrap();
    } else {
        trace!("Skipping code generation due to CLI flag");
    }
}
