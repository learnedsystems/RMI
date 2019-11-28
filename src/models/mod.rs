// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
mod balanced_radix;
mod bottom_up_plr;
mod cubic_spline;
mod histogram;
mod linear;
mod linear_spline;
mod normal;
mod pgm;
mod radix;
mod stdlib;
mod utils;

pub use balanced_radix::BalancedRadixModel;
pub use bottom_up_plr::BottomUpPLR;
pub use cubic_spline::CubicSplineModel;
pub use histogram::EquidepthHistogramModel;
pub use linear::LinearModel;
pub use linear::LogLinearModel;
pub use linear_spline::LinearSplineModel;
pub use normal::LogNormalModel;
pub use normal::NormalModel;
pub use pgm::PGM;
pub use radix::RadixModel;
pub use stdlib::StdFunctions;

use std::collections::HashSet;

#[derive(Clone)]
pub enum ModelData {
    IntKeyToIntPos(Vec<(u64, u64)>),
    #[allow(dead_code)]
    FloatKeyToIntPos(Vec<(f64, u64)>),
    #[allow(dead_code)]
    IntKeyToFloatPos(Vec<(u64, f64)>),
    FloatKeyToFloatPos(Vec<(f64, f64)>),
}

#[cfg(test)]
macro_rules! vec_to_ii {
    ($x:expr) => {
        ($x).into_iter()
            .map(|(x, y)| (x as u64, y as u64))
            .collect()
    };
}

macro_rules! extract_and_convert_tuple {
    ($vec: expr, $idx: expr, $type1:ty, $type2:ty) => {{
        let (x, y) = $vec[$idx];
        (x as $type1, y as $type2)
    }};
}

macro_rules! scale_to_max {
    ($max_val:expr, $num_rows:expr, $y_type:ty, $vec:expr) => {
        for i in 0..($vec).len() {
            let (x, y) = ($vec)[i];
            let y_val = ($max_val) as f64 * (y as f64) / ($num_rows as f64);
            ($vec)[i] = (x, y_val as $y_type);
        }
    };
}

macro_rules! define_iterator_type {
    ($name: tt, $type1: ty, $type2: ty) => {
        pub struct $name<'a> {
            data: &'a ModelData,
            idx: usize,
        }

        impl<'a> $name<'a> {
            fn new(data: &'a ModelData) -> $name<'a> {
                return $name { data: data, idx: 0 };
            }
        }

        impl<'a> Iterator for $name<'a> {
            type Item = ($type1, $type2);

            fn next(&mut self) -> Option<Self::Item> {
                if self.idx >= self.data.len() {
                    return None;
                }

                let itm = match self.data {
                    ModelData::FloatKeyToFloatPos(data) => {
                        extract_and_convert_tuple!(data, self.idx, $type1, $type2)
                    }
                    ModelData::FloatKeyToIntPos(data) => {
                        extract_and_convert_tuple!(data, self.idx, $type1, $type2)
                    }
                    ModelData::IntKeyToIntPos(data) => {
                        extract_and_convert_tuple!(data, self.idx, $type1, $type2)
                    }
                    ModelData::IntKeyToFloatPos(data) => {
                        extract_and_convert_tuple!(data, self.idx, $type1, $type2)
                    }
                };
                self.idx += 1;

                return Some(itm);
            }
        }
    };
}

define_iterator_type!(ModelDataFFIterator, f64, f64);
define_iterator_type!(ModelDataIIIterator, u64, u64);
//define_iterator_type!(ModelDataFIIterator, f64, u64);
//define_iterator_type!(ModelDataIFIterator, u64, f64);

impl ModelData {
    pub fn iter_float_float(&self) -> ModelDataFFIterator {
        return ModelDataFFIterator::new(&self);
    }
    pub fn iter_int_int(&self) -> ModelDataIIIterator {
        return ModelDataIIIterator::new(&self);
    }
    //pub fn iter_float_int(&self) -> ModelDataFIIterator { return ModelDataFIIterator::new(&self); }
    //pub fn iter_int_float(&self) -> ModelDataIFIterator { return ModelDataIFIterator::new(&self); }

    pub fn empty() -> ModelData {
        return ModelData::FloatKeyToFloatPos(vec![]);
    }

    #[cfg(test)]
    fn as_int_int(self) -> Vec<(u64, u64)> {
        return match self {
            ModelData::FloatKeyToFloatPos(data) => vec_to_ii!(data),
            ModelData::FloatKeyToIntPos(data) => vec_to_ii!(data),
            ModelData::IntKeyToFloatPos(data) => vec_to_ii!(data),
            ModelData::IntKeyToIntPos(data) => data,
        };
    }

    pub fn len(&self) -> usize {
        return match self {
            ModelData::FloatKeyToFloatPos(data) => data.len(),
            ModelData::FloatKeyToIntPos(data) => data.len(),
            ModelData::IntKeyToFloatPos(data) => data.len(),
            ModelData::IntKeyToIntPos(data) => data.len(),
        };
    }

    pub fn scale_targets_to(&mut self, max_val: u64, num_rows: usize) {
        match self {
            ModelData::FloatKeyToFloatPos(data) => scale_to_max!(max_val, num_rows, f64, data),
            ModelData::FloatKeyToIntPos(data) => scale_to_max!(max_val, num_rows, u64, data),
            ModelData::IntKeyToFloatPos(data) => scale_to_max!(max_val, num_rows, f64, data),
            ModelData::IntKeyToIntPos(data) => scale_to_max!(max_val, num_rows, u64, data),
        };
    }

    pub fn get(&self, idx: usize) -> (f64, f64) {
        return match self {
            ModelData::FloatKeyToFloatPos(data) => data[idx],
            ModelData::FloatKeyToIntPos(data) => (data[idx].0, data[idx].1 as f64),
            ModelData::IntKeyToFloatPos(data) => (data[idx].0 as f64, data[idx].1),
            ModelData::IntKeyToIntPos(data) => (data[idx].0 as f64, data[idx].1 as f64),
        };
    }
}

pub enum ModelInput {
    Int(u64),
    Float(f64),
}

impl ModelInput {
    fn as_float(&self) -> f64 {
        return match self {
            ModelInput::Int(x) => *x as f64,
            ModelInput::Float(x) => *x,
        };
    }

    fn as_int(&self) -> u64 {
        return match self {
            ModelInput::Int(x) => *x,
            ModelInput::Float(x) => *x as u64,
        };
    }
}

impl From<u64> for ModelInput {
    fn from(i: u64) -> Self {
        ModelInput::Int(i)
    }
}

impl From<f64> for ModelInput {
    fn from(f: f64) -> Self {
        ModelInput::Float(f)
    }
}

pub enum ModelDataType {
    Int,
    Float,
}

impl ModelDataType {
    pub fn c_type(&self) -> &'static str {
        match self {
            ModelDataType::Int => "uint64_t",
            ModelDataType::Float => "double",
        }
    }
}

#[derive(Debug)]
pub enum ModelParam {
    Int(u64),
    Float(f64),
    ShortArray(Vec<u16>),
    IntArray(Vec<u64>),
    FloatArray(Vec<f64>),
}

impl ModelParam {
    // size in bytes
    pub fn size(&self) -> usize {
        match self {
            ModelParam::Int(_) => 8,
            ModelParam::Float(_) => 8,
            ModelParam::ShortArray(a) => 2 * a.len(),
            ModelParam::IntArray(a) => 8 * a.len(),
            ModelParam::FloatArray(a) => 8 * a.len(),
        }
    }

    pub fn c_type(&self) -> &'static str {
        match self {
            ModelParam::Int(_) => "uint64_t",
            ModelParam::Float(_) => "double",
            ModelParam::ShortArray(_) => "short",
            ModelParam::IntArray(_) => "uint64_t",
            ModelParam::FloatArray(_) => "double",
        }
    }

    pub fn c_type_mod(&self) -> &'static str {
        match self {
            ModelParam::Int(_) => "",
            ModelParam::Float(_) => "",
            ModelParam::ShortArray(_) => "[]",
            ModelParam::IntArray(_) => "[]",
            ModelParam::FloatArray(_) => "[]",
        }
    }

    pub fn c_val(&self) -> String {
        match self {
            ModelParam::Int(v) => format!("{}UL", v),
            ModelParam::Float(v) => {
                let mut tmp = format!("{:.}", v);
                if !tmp.contains('.') {
                    tmp.push_str(".0");
                }
                return tmp;
            }
            ModelParam::ShortArray(arr) => {
                let itms: Vec<String> = arr.iter().map(|i| format!("{}", i)).collect();
                return format!("{{ {} }}", itms.join(", "));
            }
            ModelParam::IntArray(arr) => {
                let itms: Vec<String> = arr.iter().map(|i| format!("{}UL", i)).collect();
                return format!("{{ {} }}", itms.join(", "));
            }
            ModelParam::FloatArray(arr) => {
                let itms: Vec<String> = arr
                    .iter()
                    .map(|i| format!("{:.}", i))
                    .map(|s| if !s.contains('.') { s + ".0" } else { s })
                    .collect();
                return format!("{{ {} }}", itms.join(", "));
            }
        }
    }

    pub fn as_float(&self) -> f64 {
        match self {
            ModelParam::Int(v) => *v as f64,
            ModelParam::Float(v) => *v,
            ModelParam::ShortArray(_) => panic!("Cannot treat a short array parameter as a float"),
            ModelParam::IntArray(_) => panic!("Cannot treat an int array parameter as a float"),
            ModelParam::FloatArray(_) => panic!("Cannot treat an float array parameter as a float"),
        }
    }
}

impl From<usize> for ModelParam {
    fn from(i: usize) -> Self {
        ModelParam::Int(i as u64)
    }
}

impl From<u64> for ModelParam {
    fn from(i: u64) -> Self {
        ModelParam::Int(i)
    }
}

impl From<u8> for ModelParam {
    fn from(i: u8) -> Self {
        ModelParam::Int(u64::from(i))
    }
}

impl From<f64> for ModelParam {
    fn from(f: f64) -> Self {
        ModelParam::Float(f)
    }
}

impl From<Vec<u16>> for ModelParam {
    fn from(f: Vec<u16>) -> Self {
        ModelParam::ShortArray(f)
    }
}

impl From<Vec<u64>> for ModelParam {
    fn from(f: Vec<u64>) -> Self {
        ModelParam::IntArray(f)
    }
}

impl From<Vec<f64>> for ModelParam {
    fn from(f: Vec<f64>) -> Self {
        ModelParam::FloatArray(f)
    }
}

pub enum ModelRestriction {
    None,
    MustBeTop,
    MustBeBottom,
}

pub trait Model {
    fn predict_to_float(&self, inp: ModelInput) -> f64 {
        return self.predict_to_int(inp) as f64;
    }

    fn predict_to_int(&self, inp: ModelInput) -> u64 {
        return f64::max(0.0, self.predict_to_float(inp).floor()) as u64;
    }

    fn input_type(&self) -> ModelDataType;
    fn output_type(&self) -> ModelDataType;

    fn params(&self) -> Vec<ModelParam>;

    fn code(&self) -> String;
    fn function_name(&self) -> String;

    fn standard_functions(&self) -> HashSet<StdFunctions> {
        return HashSet::new();
    }

    fn needs_bounds_check(&self) -> bool {
        return true;
    }
    fn restriction(&self) -> ModelRestriction {
        return ModelRestriction::None;
    }
    fn error_bound(&self) -> Option<u64> {
        return None;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scale() {
        let mut v = ModelData::IntKeyToIntPos(vec![(0, 0), (1, 1), (3, 2), (100, 3)]);

        v.scale_targets_to(50, 4);

        let results = v.as_int_int();
        assert_eq!(results[0].1, 0);
        assert_eq!(results[1].1, 12);
        assert_eq!(results[2].1, 25);
        assert_eq!(results[3].1, 37);
    }

    #[test]
    fn test_iter() {
        let data = vec![(0, 1), (1, 2), (3, 3), (100, 4)];

        let v = ModelData::IntKeyToIntPos(data.clone());

        let iterated: Vec<(u64, u64)> = v.iter_int_int().collect();
        assert_eq!(data, iterated);
    }
}
