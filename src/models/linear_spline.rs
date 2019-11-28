// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::*;

fn linear_splines(data: &ModelData) -> (f64, f64) {
    if data.len() == 0 {
        return (0.0, 1.0);
    }

    if data.len() == 1 {
        return (data.get(0).1, 0.0);
    }

    let first_pt = data.get(0);
    let last_pt = data.get(data.len() - 1);

    if first_pt.0 == last_pt.0 {
        // data is all duplicates!
        return (data.get(0).1, 0.0);
    }

    let slope = (first_pt.1 - last_pt.1) / (first_pt.0 - last_pt.0);
    let intercept = first_pt.1 - slope * first_pt.0;

    return (intercept, slope);
}

pub struct LinearSplineModel {
    params: (f64, f64),
}

impl LinearSplineModel {
    pub fn new(data: &ModelData) -> LinearSplineModel {
        return LinearSplineModel {
            params: linear_splines(data),
        };
    }
}

impl Model for LinearSplineModel {
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
    fn test_linear_spline1() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2), (2, 3), (3, 8)]);

        let lin_mod = LinearSplineModel::new(&md);

        assert_eq!(lin_mod.predict_to_int(1.into()), 2);
        assert_eq!(lin_mod.predict_to_int(3.into()), 8);
    }

    #[test]
    fn test_linear_spline_single() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2)]);

        let lin_mod = LinearSplineModel::new(&md);

        assert_eq!(lin_mod.predict_to_int(1.into()), 2);
    }

    #[test]
    fn test_empty() {
        LinearSplineModel::new(&ModelData::empty());
    }

}
