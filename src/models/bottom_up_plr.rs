// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::utils::plr;
use crate::models::*;
use log::*;

const NUM_RADIX_BITS: usize = 20;

fn bottom_up_plr(data: &ModelData) -> (Vec<u16>, Vec<u64>, Vec<f64>) {
    if data.len() == 0 {
        return (Vec::new(), Vec::new(), Vec::new());
    }

    let mut delta = 1.0;
    let (mut points, mut coeffs) = plr(data, delta, false);

    while points.len() > 524_288 {
        delta *= 2.0;
        let (p, c) = plr(data, delta, false);
        points = p;
        coeffs = c;
    }

    info!("PLR found {} segments at delta = {}", points.len(), delta);

    // build the radix index
    let mut radix_index: Vec<u16> = vec![0; 1 << NUM_RADIX_BITS];
    let mut search_idx = 0;

    for pt in 0..radix_index.len() {
        let ones = !(0 as u64) >> NUM_RADIX_BITS;
        let desired_radix = (pt as u64) << (64 - NUM_RADIX_BITS);
        assert!(pt == 0 || desired_radix > 0);

        let key = ones | desired_radix;
        assert_eq!(key >> (64 - NUM_RADIX_BITS), pt as u64);
        assert!(key == std::u64::MAX || (key + 1) >> (64 - NUM_RADIX_BITS) == pt as u64 + 1);

        while search_idx < points.len() && points[search_idx] <= key {
            search_idx += 1;
        }
        assert!(search_idx > 0);
        radix_index[pt] = search_idx as u16;
        assert!(search_idx == points.len() || points[search_idx] > key);
    }

    return (radix_index, points, coeffs);
}

pub struct BottomUpPLR {
    radix: Vec<u16>,
    points: Vec<u64>,
    coeffs: Vec<f64>,
}

impl BottomUpPLR {
    pub fn new(data: &ModelData) -> BottomUpPLR {
        let (radix, points, coeffs) = bottom_up_plr(data);
        return BottomUpPLR {
            radix,
            points,
            coeffs,
        };
    }
}

impl Model for BottomUpPLR {
    fn predict_to_float(&self, inp: ModelInput) -> f64 {
        let val = inp.as_int(); //4098767424329; //inp.as_int();

        // TODO we could accelerate training time by using the radix index here
        let bs_result = self.points.binary_search(&val);

        let mut line_index = match bs_result {
            Ok(idx) => idx,
            Err(idx) => idx - 1,
        };

        if line_index == self.points.len() {
            line_index -= 1;
        }

        assert!(
            self.points[line_index] <= val,
            "previous segment (idx {}) stops at {} and val is {}",
            line_index,
            self.points[line_index - 1],
            val
        );
        assert!(line_index == self.points.len() - 1 || self.points[line_index + 1] > val);

        // verify that the radix table would have given valid bounds
        let radix_hint = val >> (64 - NUM_RADIX_BITS);
        let radix_ub = self.radix[radix_hint as usize] as usize;
        let radix_lb = if radix_hint == 0 {
            0
        } else {
            self.radix[radix_hint as usize - 1] as usize - 1
        };
        assert!(
            radix_lb <= line_index,
            "radix key: {} radix lb: {}, radix ub: {}, correct: {}, key: {}, value: {}",
            radix_hint,
            radix_lb,
            radix_ub,
            line_index,
            val,
            self.points[line_index]
        );
        assert!(
            radix_ub > line_index,
            "radix key: {} radix lb: {} radix ub: {}, correct: {}, key: {}, value: {}",
            radix_hint,
            radix_lb,
            radix_ub,
            line_index,
            val,
            self.points[line_index]
        );

        let a = self.coeffs[2 * line_index];
        let b = self.coeffs[2 * line_index + 1];
        let pred = (val as f64) * a + b;
        //println!("{} from {}", pred, line_index);
        //panic!();
        return pred;
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![
            self.points.len().into(),
            self.radix.clone().into(),
            self.points.clone().into(),
            self.coeffs.clone().into(),
        ];
    }

    fn code(&self) -> String {
        return format!(
            "
inline uint64_t plr(const uint64_t size, 
                    const short radix[],
                    const uint64_t pivots[], const double coeffs[], uint64_t key) {{
    uint64_t key_radix = key >> (64 - {});
    unsigned int radix_ub = radix[key_radix];
    unsigned int radix_lb = (key_radix == 0 ? 0 : radix[key_radix - 1] - 1);
    uint64_t li = bs_upper_bound(pivots + radix_lb, radix_ub - radix_lb, key) + radix_lb - 1;

    double alpha = coeffs[2*li];
    double beta = coeffs[2*li + 1];
    return alpha * (double)key + beta;
}}
",
            NUM_RADIX_BITS
        );
    }

    fn standard_functions(&self) -> HashSet<StdFunctions> {
        let mut to_r = HashSet::new();
        to_r.insert(StdFunctions::BinarySearch);
        return to_r;
    }

    fn function_name(&self) -> String {
        return String::from("plr");
    }
    fn restriction(&self) -> ModelRestriction {
        return ModelRestriction::MustBeBottom;
    }
}
