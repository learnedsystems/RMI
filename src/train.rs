// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::*;
use log::*;

pub struct TrainedRMI {
    pub model_avg_error: f64,
    pub model_max_error: u64,
    pub model_max_error_idx: usize,
    pub last_layer_max_l1s: Vec<u64>,
    pub rmi: Vec<Vec<Box<dyn Model>>>,
}

fn train_model(model_type: &str, data: &ModelData) -> Box<dyn Model> {
    let model: Box<dyn Model> = match model_type {
        "linear" => Box::new(LinearModel::new(data)),
        "linear_spline" => Box::new(LinearSplineModel::new(data)),
        "cubic" => Box::new(CubicSplineModel::new(data)),
        "loglinear" => Box::new(LogLinearModel::new(data)),
        "normal" => Box::new(NormalModel::new(data)),
        "lognormal" => Box::new(LogNormalModel::new(data)),
        "radix" => Box::new(RadixModel::new(data)),
        "bradix" => Box::new(BalancedRadixModel::new(data)),
        "histogram" => Box::new(EquidepthHistogramModel::new(data)),
        "plr" => Box::new(BottomUpPLR::new(data)),
        "pgm" => Box::new(PGM::new(data)),
        _ => panic!("Unknown model type: {}", model_type),
    };

    return model;
}

fn validate(model_spec: &[String]) {
    let num_layers = model_spec.len();
    let empty_data = ModelData::empty();

    for (idx, model) in model_spec.iter().enumerate() {
        let restriction = train_model(model, &empty_data).restriction();

        match restriction {
            ModelRestriction::None => {}
            ModelRestriction::MustBeTop => {
                assert_eq!(
                    idx, 0,
                    "if used, model type {} must be the root model",
                    model
                );
            }
            ModelRestriction::MustBeBottom => {
                assert_eq!(
                    idx,
                    num_layers - 1,
                    "if used, model type {} must be the bottommost model",
                    model
                );
            }
        }
    }
}

pub fn train(data: ModelData, model_spec: &str, branch_factor: u64) -> TrainedRMI {
    let (model_list, last_model): (Vec<String>, String) = {
        let mut all_models: Vec<String> = model_spec.split(',').map(String::from).collect();
        validate(&all_models);
        let last = all_models.pop().unwrap();
        (all_models, last)
    };

    let mut rmi: Vec<Vec<Box<dyn Model>>> = Vec::new();
    let mut data_partitions = vec![data];
    let num_rows = data_partitions[0].len();

    let mut current_model_count = 1;
    for model_type in model_list {
        info!("Training {} model layer", model_type);
        // data_partition contains all of our data partitioned into groups
        // based on the previous RMI layer's output
        let next_layer_size = current_model_count * branch_factor;
        let mut next_layer_data =
            vec![Vec::with_capacity(num_rows / next_layer_size as usize); next_layer_size as usize];
        let mut models: Vec<Box<dyn Model>> = Vec::with_capacity(next_layer_size as usize);

        for model_data in data_partitions.into_iter() {
            let mut tmp_data = model_data.clone();
            // not at the last layer -- rescale
            tmp_data.scale_targets_to(next_layer_size, num_rows);

            let model = train_model(model_type.as_str(), &tmp_data);

            for (x, y) in model_data.iter_int_int() {
                let model_pred = model.predict_to_int(x.into());
                assert!(model.needs_bounds_check() || model_pred < next_layer_size);
                let target = u64::min(next_layer_size - 1, model_pred) as usize;

                next_layer_data[target].push((x, y));
            }

            models.push(model);
        }

        data_partitions = next_layer_data
            .into_iter()
            .map(ModelData::IntKeyToIntPos)
            .collect();

        current_model_count *= branch_factor;
        rmi.push(models);
    }

    info!("Training last level {} model", last_model);
    let mut last_layer = Vec::new();
    let mut last_layer_max_l1s: Vec<u64> = Vec::new();
    let mut model_avg_error = 0.0;
    let mut model_max_error = 0;
    let mut model_max_error_idx = 0;

    let mut n = 1;
    for (midx, model_data) in data_partitions.into_iter().enumerate() {
        let last_model = train_model(last_model.as_str(), &model_data);
        let mut max_error = 0;
        for (idx, (x, y)) in model_data.iter_int_int().enumerate() {
            let pred = last_model.predict_to_int(x.into());
            let err = u64::max(y, pred) - u64::min(y, pred);

            if let Some(bound) = last_model.error_bound() {
                if err > bound {
                    warn!("Precision issue: model reports max bound of {}, \
but an error of {} was observed on input {} at index {}. Prediction: {} Actual: {}",
                        bound, err, x, idx, pred, y);
                }
            }

            max_error = u64::max(max_error, err);
            model_avg_error += ((max_error as f32) - model_avg_error) / (n as f32);
            n += 1;
        }
        if max_error > model_max_error {
            model_max_error = max_error;
            model_max_error_idx = midx;
        }

        last_layer.push(last_model);
        last_layer_max_l1s.push(max_error);
    }
    rmi.push(last_layer);

    return TrainedRMI {
        model_avg_error: model_avg_error as f64,
        model_max_error,
        model_max_error_idx,
        last_layer_max_l1s,
        rmi,
    };
}
