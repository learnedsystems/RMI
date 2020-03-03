// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::*;
use log::*;
use rayon::iter::ParallelBridge;
use rayon::prelude::ParallelIterator;

pub struct TrainedRMI {
    pub model_avg_error: f64,
    pub model_avg_log2_error: f64,
    pub model_max_error: u64,
    pub model_max_error_idx: usize,
    pub last_layer_max_l1s: Vec<u64>,
    pub rmi: Vec<Vec<Box<dyn Model>>>,
}

fn train_model(model_type: &str, data: &ModelDataContainer) -> Box<dyn Model> {
    let model: Box<dyn Model> = match model_type {
        "linear" => Box::new(LinearModel::new(data)),
        "linear_spline" => Box::new(LinearSplineModel::new(data)),
        "cubic" => Box::new(CubicSplineModel::new(data)),
        "loglinear" => Box::new(LogLinearModel::new(data)),
        "normal" => Box::new(NormalModel::new(data)),
        "lognormal" => Box::new(LogNormalModel::new(data)),
        "radix" => Box::new(RadixModel::new(data)),
        "radix8" => Box::new(RadixTable::new(data, 8)),
        "radix18" => Box::new(RadixTable::new(data, 18)),
        "radix22" => Box::new(RadixTable::new(data, 22)),
        "radix26" => Box::new(RadixTable::new(data, 26)),
        "radix28" => Box::new(RadixTable::new(data, 28)),
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
    let empty_container = ModelDataContainer::new(empty_data);

    for (idx, model) in model_spec.iter().enumerate() {
        let restriction = train_model(model, &empty_container).restriction();

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

fn train_two_layer(data: ModelData,
                   layer1_model: &str, layer2_model: &str,
                   num_leaf_models: u64) -> TrainedRMI {
    validate(&[String::from(layer1_model), String::from(layer2_model)]);

    let num_rows = data.len();
    let mut md_container = ModelDataContainer::new(data);

    let dummy_md = ModelDataContainer::new(ModelData::empty());

    info!("Training top-level {} model layer", layer1_model);
    md_container.set_scale(num_leaf_models as f64 / num_rows as f64);
    let top_model = train_model(layer1_model, &md_container);
    let mut leaf_models: Vec<Box<dyn Model>> = Vec::new();

    let mut second_layer_data = Vec::with_capacity(num_rows / num_leaf_models as usize);
    let mut last_target = 0;
    
    info!("Training second-level {} model layer (num models = {})", layer2_model, num_leaf_models);
    md_container.set_scale(1.0);

    for (x, y) in md_container.iter_int_int() {
        let model_pred = top_model.predict_to_int(x.into());
        assert!(top_model.needs_bounds_check() || model_pred < num_leaf_models,
                "model produced result {} which is out of bounds with {} leaf models",
                model_pred, num_leaf_models);
        let target = u64::min(num_leaf_models - 1, model_pred) as usize;
        
        assert!(target >= last_target);
        
        if target > last_target {
            // this is the first datapoint for the next leaf model.
            // train the previous leaf model.
            let container = ModelDataContainer::new(ModelData::IntKeyToIntPos(second_layer_data));
            let leaf_model = train_model(layer2_model, &container);
            leaf_models.push(leaf_model);
            
            
            // leave empty models for any we skipped.
            for _skipped_idx in (last_target+1)..target {
                leaf_models.push(train_model(layer2_model, &dummy_md));
            }
            assert_eq!(leaf_models.len(), target);
            
            second_layer_data = Vec::new();
        }
        
        second_layer_data.push((x, y));
        last_target = target;
    }

    // train the last remaining model
    let container = ModelDataContainer::new(ModelData::IntKeyToIntPos(second_layer_data));
    let leaf_model = train_model(layer2_model, &container);
    leaf_models.push(leaf_model);
    
    // add models at the end with nothing mapped into them
    for _skipped_idx in (last_target+1)..num_leaf_models as usize {
        leaf_models.push(train_model(layer2_model, &dummy_md));
    }
    assert_eq!(num_leaf_models as usize, leaf_models.len());


    info!("Computing last level errors...");
    let mut last_layer_max_l1s = vec![0 ; num_leaf_models as usize];


    // evaluate model, compute last level errors
    for (x, y) in md_container.iter_int_int() {
        let leaf_idx = top_model.predict_to_int(x.into());
        let target = u64::min(num_leaf_models - 1, leaf_idx) as usize;

        let pred = leaf_models[target].predict_to_int(x.into());
        let err = u64::max(y, pred) - u64::min(y, pred);
        last_layer_max_l1s[target] = u64::max(last_layer_max_l1s[target], err);
    }

    info!("Evaluating two-layer RMI...");
    let (m_idx, m_err) = last_layer_max_l1s
        .iter().enumerate()
        .max_by_key(|(_idx, &x)| x).unwrap();
    
    let model_max_error = *m_err;
    let model_max_error_idx = m_idx;

    let model_avg_error: f64 = last_layer_max_l1s
        .iter().sum::<u64>() as f64 / last_layer_max_l1s.len() as f64;
    
    let model_avg_log2_error: f64 =last_layer_max_l1s
        .iter().map(|&x| ((2*x + 2) as f64).log2()).sum::<f64>() / last_layer_max_l1s.len() as f64;

    
    return TrainedRMI {
        model_avg_error,
        model_avg_log2_error,
        model_max_error,
        model_max_error_idx,
        last_layer_max_l1s,
        rmi: vec![vec![top_model], leaf_models]
    };

}

pub fn train(data: ModelData, model_spec: &str, branch_factor: u64) -> TrainedRMI {
    let (model_list, last_model): (Vec<String>, String) = {
        let mut all_models: Vec<String> = model_spec.split(',').map(String::from).collect();
        validate(&all_models);
        let last = all_models.pop().unwrap();
        (all_models, last)
    };

    if model_list.len() == 1 {
        return train_two_layer(data, &model_list[0], &last_model, branch_factor);
    }

    let mut rmi: Vec<Vec<Box<dyn Model>>> = Vec::new();
    let mut data_partitions = vec![data];
    let num_rows = data_partitions[0].len();

    let mut current_model_count = 1;
    for (_layer_idx, model_type) in model_list.into_iter().enumerate() {
        info!("Training {} model layer", model_type);
        // data_partition contains all of our data partitioned into groups
        // based on the previous RMI layer's output
        let next_layer_size = current_model_count * branch_factor;
        let mut next_layer_data =
            vec![Vec::with_capacity(num_rows / next_layer_size as usize); next_layer_size as usize];
        let mut models: Vec<Box<dyn Model>> = Vec::with_capacity(next_layer_size as usize);

        for model_data in data_partitions.into_iter() {
            let mut md_container = ModelDataContainer::new(model_data);

            // not at the last layer -- rescale
            md_container.set_scale(next_layer_size as f64 / num_rows as f64);
            let model = train_model(model_type.as_str(), &md_container);

            // rescale back for next layer
            md_container.set_scale(1.0);

            for (x, y) in md_container.iter_int_int() {
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
    let mut model_avg_error: f64 = 0.0;
    let mut model_avg_log2_error: f64 = 0.0;
    let mut model_max_error = 0;
    let mut model_max_error_idx = 0;

    let mut n = 1;
    for (midx, model_data) in data_partitions.into_iter().enumerate() {
        let md_container = ModelDataContainer::new(model_data);
        let last_model = train_model(last_model.as_str(), &md_container);
        let mut max_error = 0;
        
        for (idx, (x, y)) in md_container.iter_int_int().enumerate() {
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
            model_avg_error += ((max_error as f64) - model_avg_error) / (n as f64);
            let log2_error = ((2 * max_error + 2) as f64).log2();
            model_avg_log2_error += (log2_error - model_avg_log2_error) / (n as f64);
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
        model_avg_error,
        model_avg_log2_error,
        model_max_error,
        model_max_error_idx,
        last_layer_max_l1s,
        rmi,
    };
}
