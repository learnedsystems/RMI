// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::utils::plr;
use crate::models::*;
use log::*;
use superslice::*;

const FIRST_LAYER_DELTA: f64 = 64.0;
const OTHER_LAYER_DELTA: f64 = 4.0;
const SMALLEST_LAYER_SIZE: usize = 32;

const SEARCH_ERR_MARGIN: usize = (2.0 * OTHER_LAYER_DELTA) as usize;

fn pgm(data: &ModelData) -> (Vec<Vec<u64>>, Vec<Vec<f64>>) {
    if data.len() == 0 {
        return (Vec::new(), Vec::new());
    }

    // we'll build the PGM bottom up, starting with the base level.
    // use a delta of 64, as per the fastest (and biggest) PGM in the paper
    let (bottom_pts, bottom_coefs) = plr(data, FIRST_LAYER_DELTA, false);

    let mut pts: Vec<Vec<u64>> = vec![bottom_pts];
    let mut coefs: Vec<Vec<f64>> = vec![bottom_coefs];

    // build each level using delta = OTHER_LAYER_DELTA on the previous level,
    // stopping when we have a layer with <= SMALLEST_LAYER_SIZE points.
    while pts.last().unwrap().len() > SMALLEST_LAYER_SIZE {
        // first, we construct the dataset
        let mut d = Vec::new();

        for (idx, &itm) in pts.last().unwrap().iter().enumerate() {
            d.push((itm, idx as u64));
        }

        let md = ModelData::IntKeyToIntPos(d);

        let (p, c) = plr(&md, OTHER_LAYER_DELTA, true);
        pts.push(p);
        coefs.push(c);
    }

    pts.reverse();
    coefs.reverse();

    let layer_sizes: Vec<usize> = pts.iter().map(|v| v.len()).collect();
    info!("PGM model trained, layer sizes: {:?}", layer_sizes);

    return (pts, coefs);
}

pub struct PGM {
    points: Vec<Vec<u64>>,
    coeffs: Vec<Vec<f64>>,
}

impl PGM {
    pub fn new(data: &ModelData) -> PGM {
        let (points, coeffs) = pgm(data);
        return PGM { points, coeffs };
    }
}

impl Model for PGM {
    fn predict_to_float(&self, inp: ModelInput) -> f64 {
        // we convert the model input to a float first and then to a u64
        // to ensure we get the same u64 -> float mapping that will happen
        // in C. Example: 42285439947654605 -> 42285439947654610
        let ukey = inp.as_float() as u64;

        // when code is generated, we need to traverse the whole tree,
        // but for training and evaluating the model's accuracy,
        // we can simply use the last layer.
        let layer = self.points.last().unwrap();
        let coeffs = self.coeffs.last().unwrap();
        let idx = layer.upper_bound(&ukey) - 1;
        /*println!("Predicting {} (original: {}) from model {}",
        ukey, inp.as_int(), idx);*/
        let pos = f64::max(0.0, coeffs[2 * idx] * inp.as_float() + coeffs[2 * idx + 1]) as usize;
        return pos as f64;
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    fn params(&self) -> Vec<ModelParam> {
        let layer_sizes: Vec<u64> = self.points.iter().map(|v| v.len() as u64).collect();

        let flat_points: Vec<u64> = self.points.iter().flat_map(|v| v).copied().collect();

        let flat_coeffs: Vec<f64> = self.coeffs.iter().flat_map(|v| v).copied().collect();

        return vec![
            ModelParam::IntArray(layer_sizes),
            ModelParam::IntArray(flat_points),
            ModelParam::FloatArray(flat_coeffs),
        ];
    }

    fn code(&self) -> String {
        let num_layers = self.points.len();
        return format!(
            "
#include <cstdint>

#define MAX(x,  y)   (((x) > (y)) ? (x) : (y))
#define MIN(x,  y)   (((x) < (y)) ? (x) : (y))



const unsigned int NUM_LAYERS = {};
const double LAYER_DELTA = {};

inline uint64_t search(const uint64_t points[], uint64_t lsize,
                       int64_t pred, uint64_t key) {{
    uint64_t start = (pred <= LAYER_DELTA ? 0 : pred - LAYER_DELTA);
    start = MIN(start, lsize - 1);
    uint64_t stop = pred + LAYER_DELTA;


    while (start > 0 && points[start] > key) start /= 2;
    while (stop < lsize && points[stop] < key) stop *= 2;
    stop = MIN(pred + LAYER_DELTA, lsize);


    uint64_t res = bs_upper_bound(points + start, stop - start, key) + start;
    return (res == 0 ? 0 : res - 1);
}}

uint64_t pgm(const uint64_t layer_sizes[], 
           const uint64_t f_points[], const double f_coeffs[],
           const double key) {{

    uint64_t ukey = (uint64_t) key;
    uint64_t pos = layer_sizes[0] / 2;
    const uint64_t* points = f_points;
    const double* coeffs = f_coeffs;
    
    for (uint64_t i = 0; i < NUM_LAYERS; i++) {{

        pos = search(points, layer_sizes[i], pos, ukey);

        if (pos == layer_sizes[i] - 1) {{
            pos = (uint64_t) MAX(0.0, coeffs[2*pos] * key + coeffs[2*pos+1]);
        }} else {{
            double fa = coeffs[2*pos];
            double fb = coeffs[2*pos+1];
            double ga = coeffs[2*(pos+1)];
            double gb = coeffs[2*(pos+1)+1];
            
            double fpred = fa * key + fb;
            double gpred = ga * (double)points[pos+1] + gb;

            pos = (uint64_t) MAX(0.0, MIN(fpred, gpred));
        }}
        points += layer_sizes[i];
        coeffs += 2*layer_sizes[i];
    }}

    return pos;
}}
",
            num_layers, SEARCH_ERR_MARGIN
        );
    }

    fn standard_functions(&self) -> HashSet<StdFunctions> {
        let mut to_r = HashSet::new();
        to_r.insert(StdFunctions::BinarySearch);
        return to_r;
    }

    fn function_name(&self) -> String {
        return String::from("pgm");
    }
    fn restriction(&self) -> ModelRestriction {
        return ModelRestriction::MustBeBottom;
    }
    fn error_bound(&self) -> Option<u64> {
        return Some(2 * (FIRST_LAYER_DELTA as u64));
    }
}
