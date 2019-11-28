// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::*;

fn slr(loc_data: &ModelData) -> (f64, f64) {
    let data_size = loc_data.len();

    // special case when we have 0 or 1 items
    if data_size == 0 {
        return (0.0, 0.0);
    }

    if data_size == 1 {
        let (_, y) = loc_data.iter_float_float().next().unwrap();
        return (y, 0.0);
    }

    // compute the covariance of x and y as well as the variance of x in
    // a single pass.

    let mut mean_x = 0.0;
    let mut mean_y = 0.0;
    let mut c = 0.0;
    let mut n = 0;
    let mut m2 = 0.0;

    for (x, y) in loc_data.iter_float_float() {
        n += 1;
        let dx = x - mean_x;
        mean_x += dx / f64::from(n);
        mean_y += (y - mean_y) / f64::from(n);
        c += dx * (y - mean_y);

        let dx2 = x - mean_x;
        m2 += dx * dx2;
    }

    let cov = c / f64::from(n - 1);
    let var = m2 / f64::from(n - 1);
    assert!(var >= 0.0);

    if var == 0.0 {
        // variance is zero. pick the lowest value.
        let (_x, y) = loc_data
            .iter_float_float()
            .min_by(|(y1, _), (y2, _)| y1.partial_cmp(y2).unwrap())
            .unwrap();
        return (y, 0.0);
    }

    let beta: f64 = cov / var;
    let alpha = mean_y - beta * mean_x;

    return (alpha, beta);
}

fn loglinear_slr(data: &ModelData) -> (f64, f64) {
    // log all of the outputs, omit any item that doesn't have a valid log
    let transformed_data: Vec<(f64, f64)> = data
        .iter_float_float()
        .map(|(x, y)| (x, y.ln()))
        .filter(|(_, y)| y.is_finite())
        .collect();

    // TODO this currently creates a copy of the data and then calls
    // slr... we can probably do better by moving the log into the slr.
    return slr(&ModelData::FloatKeyToFloatPos(transformed_data));
}

pub struct LinearModel {
    params: (f64, f64),
}

impl LinearModel {
    pub fn new(data: &ModelData) -> LinearModel {
        return LinearModel { params: slr(&data) };
    }
}

impl Model for LinearModel {
    fn predict_to_float(&self, inp: ModelInput) -> f64 {
        let (alpha, beta) = self.params;
        return alpha + beta * inp.as_float();
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![self.params.0.into(), self.params.1.into()];
    }

    fn code(&self) -> String {
        return String::from(
            "
inline double linear(double alpha, double beta, double inp) {
    return alpha + beta * inp;
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("linear");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear1() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2), (2, 3), (3, 4)]);

        let lin_mod = LinearModel::new(&md);

        assert_eq!(lin_mod.predict_to_int(1.into()), 2);
        assert_eq!(lin_mod.predict_to_int(6.into()), 7);
    }

    #[test]
    fn test_linear_single() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2)]);

        let lin_mod = LinearModel::new(&md);

        assert_eq!(lin_mod.predict_to_int(1.into()), 2);
    }

    #[test]
    fn test_empty() {
        LinearModel::new(&ModelData::empty());
    }

}

pub struct LogLinearModel {
    params: (f64, f64),
}

fn exp1(inp: f64) -> f64 {
    let mut x = inp;
    x = 1.0 + x / 64.0;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    x *= x;
    return x;
}

impl LogLinearModel {
    pub fn new(data: &ModelData) -> LogLinearModel {
        return LogLinearModel {
            params: loglinear_slr(&data),
        };
    }
}

impl Model for LogLinearModel {
    fn predict_to_float(&self, inp: ModelInput) -> f64 {
        let (alpha, beta) = self.params;
        return exp1(alpha + beta * inp.as_float());
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![self.params.0.into(), self.params.1.into()];
    }

    fn code(&self) -> String {
        return String::from(
            "
inline double loglinear(double alpha, double beta, double inp) {
    return exp1(alpha + beta * inp);
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("loglinear");
    }
    fn standard_functions(&self) -> HashSet<StdFunctions> {
        let mut to_r = HashSet::new();
        to_r.insert(StdFunctions::EXP1);
        return to_r;
    }
}

#[cfg(test)]
mod loglin_tests {
    use super::*;

    #[test]
    fn test_loglinear1() {
        let md = ModelData::IntKeyToIntPos(vec![(2, 2), (3, 4), (4, 16)]);

        let loglin_mod = LogLinearModel::new(&md);

        assert_eq!(loglin_mod.predict_to_int(2.into()), 1);
        assert_eq!(loglin_mod.predict_to_int(4.into()), 13);
    }

    #[test]
    fn test_empty() {
        LogLinearModel::new(&ModelData::empty());
    }
}
