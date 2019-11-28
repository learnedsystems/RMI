// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::*;
use log::*;

fn equidepth_histogram(data: &ModelData) -> Vec<u64> {
    if data.len() == 0 {
        return Vec::new();
    }

    let mut splits: Vec<u64> = Vec::new();
    let num_bins = data.get(data.len() - 1).1 as usize;
    let items_per_bin = data.len() / num_bins;

    assert!(
        items_per_bin >= 1,
        "not enough items for equidepth histogram"
    );

    if num_bins > 2000 {
        warn!(
            "Equidepth histogram using {} bins, which is very high!",
            num_bins
        );
    } else {
        info!("Equidepth histogram using {} bins", num_bins);
    }

    for bin_idx in 0..num_bins {
        let start_idx = bin_idx * items_per_bin;
        let start_val = data.get(start_idx).0 as u64;
        splits.push(start_val);
    }

    return splits;
}

pub struct EquidepthHistogramModel {
    params: Vec<u64>,
}

impl EquidepthHistogramModel {
    pub fn new(data: &ModelData) -> EquidepthHistogramModel {
        return EquidepthHistogramModel {
            params: equidepth_histogram(data),
        };
    }
}

impl Model for EquidepthHistogramModel {
    fn predict_to_int(&self, inp: ModelInput) -> u64 {
        let val = inp.as_int();

        let mut val = match self.params.binary_search(&val) {
            Ok(val) => val,
            Err(val) => val,
        } as u64;

        val = if val == 0 { 0 } else { val - 1 };
        return val;

        /*for (idx, &split) in self.params.iter().enumerate() {
            if val <= split { return (idx - 1) as u64; }
        }

        return self.params.len() as u64 - 1;*/
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    fn params(&self) -> Vec<ModelParam> {
        let mut params = self.params.clone();
        params.insert(0, self.params.len() as u64);
        return vec![params.into()];
    }
    fn code(&self) -> String {
        return String::from(
            "
inline uint64_t ed_histogram(const uint64_t data[], uint64_t key) {
    uint64_t lb = bs_upper_bound(data + 1, *data, key);
    return (lb == 0 ? 0 : lb - 1);
}
",
        );
    }

    fn standard_functions(&self) -> HashSet<StdFunctions> {
        let mut to_r = HashSet::new();
        to_r.insert(StdFunctions::BinarySearch);
        return to_r;
    }

    fn function_name(&self) -> String {
        return String::from("ed_histogram");
    }
    fn restriction(&self) -> ModelRestriction {
        return ModelRestriction::MustBeTop;
    }
    fn needs_bounds_check(&self) -> bool {
        return false;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ed_hist1() {
        let mut test_data: Vec<(u64, u64)> = Vec::new();

        for i in 0..1000 {
            test_data.push((i * 3, i / 3));
        }

        let md = ModelData::IntKeyToIntPos(test_data);

        let ed_mod = EquidepthHistogramModel::new(&md);

        assert_eq!(ed_mod.predict_to_int((0).into()), 0);
        assert_eq!(ed_mod.predict_to_int((1 * 3).into()), 0);
        assert_eq!(ed_mod.predict_to_int((4 * 3).into()), 1);
        assert_eq!(ed_mod.predict_to_int((500 * 3).into()), 166);
        assert_eq!(ed_mod.predict_to_int((5000 * 3).into()), 332);
    }

    #[test]
    fn test_empty() {
        EquidepthHistogramModel::new(&ModelData::empty());
    }

}
