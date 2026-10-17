// < begin copyright > 
// Copyright Ryan Marcus 2020
// 
// See root directory of this project for license terms.
// 
// < end copyright > 
 

use crate::models::*;

macro_rules! scale {
    ($val: expr, $min: expr, $max: expr) => {
        ($val - $min) / ($max - $min)
    };
}

#[allow(clippy::float_cmp)]
fn cubic<T: TrainingKey>(data: &RMITrainingData<T>) -> (f64, f64, f64, f64) {
    if data.len() == 0 {
        return (0.0, 0.0, 1.0, 0.0);
    }

    if data.len() == 1 {
        return (0.0, 0.0, 0.0, data.get(0).1 as f64);
    }

    // ensure we have at least two unique values
    {
        let candidate = data.get(0).0;
        let uniq = data.iter().any(|(x, _y)| x != candidate);

        if !uniq {
            // all the same value!
            return (0.0, 0.0, 0.0, data.get(0).1 as f64);
        }
    }

    let first_pt = data.get(0);
    let last_pt = data.get(data.len() - 1);
    let (xmin, ymin) = (first_pt.0.as_float(), first_pt.1 as f64);
    let (xmax, ymax) = (last_pt.0.as_float(), last_pt.1 as f64);

    let (x1, y1) = (0.0, 0.0);
    let (x2, y2) = (1.0, 1.0);

    let mut m1 = {
        let (xn, yn) = data
            .iter()
            .find(|&(tx, _ty)| scale!(tx.as_float(), xmin, xmax) > 0.0)
            .unwrap();

        let (sxn, syn) = (scale!(xn.as_float(), xmin, xmax), scale!(yn as f64, ymin, ymax));
        (syn - y1) / (sxn - x1)
    };

    let mut m2 = {
        let (xp, yp) = (0..data.len())
            .rev()
            .map(|idx| data.get(idx))
            .find(|&(tx, _ty)| scale!(tx.as_float(), xmin, xmax) < 1.0)
            .unwrap();

        let (sxp, syp) = (scale!(xp.as_float(), xmin, xmax), scale!(yp as f64, ymin, ymax));
        (y2 - syp) / (x2 - sxp)
    };

    // keep it monotonic
    if m1.powf(2.0) + m2.powf(2.0) > 9.0 {
        let tau = 3.0 / (m1.powf(2.0) + m2.powf(2.0)).sqrt();
        m1 *= tau;
        m2 *= tau;
    }

    // from sympy, the first (a) term is:
    // '(m1 + m2 - 2)/(xmax - xmin)**3'
    let mut a = (m1 + m2 - 2.0) / (xmax - xmin).powf(3.0);

    // the second (b) term is:
    // '-(xmax*(2*m1 + m2 - 3) + xmin*(m1 + 2*m2 - 3))/(xmax - xmin)**3'
    let mut b =
        -(xmax * (2.0 * m1 + m2 - 3.0) + xmin * (m1 + 2.0 * m2 - 3.0)) / (xmax - xmin).powf(3.0);

    // the third (c) term is:
    // '(m1*xmax**2 + m2*xmin**2 + xmax*xmin*(2*m1 + 2*m2 - 6))
    //  /(xmax - xmin)**3'
    let mut c =
        (m1 * xmax.powf(2.0) + m2 * xmin.powf(2.0) + xmax * xmin * (2.0 * m1 + 2.0 * m2 - 6.0))
            / (xmax - xmin).powf(3.0);

    // the fourth (d) term is:
    // '-xmin*(m1*xmax**2 + xmax*xmin*(m2 - 3) + xmin**2)/(xmax - xmin)**3'
    let mut d = -xmin * (m1 * xmax.powf(2.0) + xmax * xmin * (m2 - 3.0) + xmin.powf(2.0))
        / (xmax - xmin).powf(3.0);

    a *= ymax - ymin;
    b *= ymax - ymin;
    c *= ymax - ymin;
    d *= ymax - ymin;
    d += ymin;
    return (a, b, c, d);
}

pub struct CubicSplineModel {
    params: (f64, f64, f64, f64),
}

impl CubicSplineModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>) -> CubicSplineModel {
        let cubic = CubicSplineModel {
            params: cubic(data),
        };

        // check our error against a linear model --
        // sometimes the slope really doesn't work out.
        let linear = LinearSplineModel::new(data);

        let mut our_error = 0.0;
        let mut lin_error = 0.0;

        for (x, y) in data.iter_model_input() {
            let c_pred = cubic.predict_to_float(&x);
            let l_pred = linear.predict_to_float(&x);

            our_error += (c_pred - (y as f64)).abs();
            lin_error += (l_pred - (y as f64)).abs();
        }

        if lin_error < our_error {
            let lp = linear.params();
            return CubicSplineModel {
                params: (0.0, 0.0, lp[1].as_float(), lp[0].as_float()),
            };
        }

        return cubic;
    }
}

impl Model for CubicSplineModel {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        let (a, b, c, d) = self.params;
        let val = inp.as_float();

        // use mul_add here so we get the same FMA behavior as we do
        // in C.
        let v1 = a.mul_add(val, b);
        let v2 = v1.mul_add(val, c);
        let v3 = v2.mul_add(val, d);
        return v3;

    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Float;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![
            self.params.0.into(),
            self.params.1.into(),
            self.params.2.into(),
            self.params.3.into(),
        ];
    }

    fn code(&self) -> String {
        return String::from(
            "
inline double cubic(double a, double b, double c, double d, double x) {
    auto v1 = std::fma(a, x, b);
    auto v2 = std::fma(v1, x, c);
    auto v3 = std::fma(v2, x, d);
    return v3;
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("cubic");
    }

    fn set_to_constant_model(&mut self, constant: u64) -> bool {
        self.params = (0.0, 0.0, 0.0, constant as f64);
        return true;
    }    

    fn prediction_ranges(&self, lo: &ModelInput, hi: &ModelInput) -> Vec<(u64, u64)> {
        let (a, b, c, d) = self.params;

        // The cubic is only monotonic between the roots of its derivative,
        // 3a x^2 + 2b x + c, so split [lo, hi] at the roots inside it.
        let mut roots = Vec::new();
        if a != 0.0 {
            let disc = 4.0 * b * b - 12.0 * a * c;
            if disc >= 0.0 {
                let q = -0.5 * (2.0 * b + (2.0 * b).signum() * disc.sqrt());
                if q != 0.0 {
                    roots.push(q / (3.0 * a));
                    roots.push(c / q);
                } else {
                    roots.push(0.0);
                }
            }
        } else if b != 0.0 {
            roots.push(-c / (2.0 * b));
        }

        let mut points = vec![*lo, *hi];
        for r in roots {
            if !(r > lo.as_float() && r < hi.as_float()) { continue; }
            match lo {
                ModelInput::Int(_) => {
                    points.push((r.floor() as u64).into());
                    points.push((r.ceil() as u64).into());
                }
                ModelInput::Float(_) => points.push(r.into())
            }
        }
        points.sort_by(|p1, p2| p1.partial_cmp(p2).unwrap());

        let mut ranges = Vec::new();
        for w in points.windows(2) {
            let f1 = self.predict_to_float(&w[0]);
            let f2 = self.predict_to_float(&w[1]);

            // Each of the three FMAs rounds once, so the computed value is
            // within about 3 * 2^-53 * (|a| x^3 + |b| x^2 + |c| x + |d|) of
            // the cubic. Widen the range by more than twice that (both ends
            // can be off), plus one. This margin for rounding is an
            // estimate (the roots above are rounded too), not proven to
            // cover every case. With a = b = 0 only the last FMA rounds and
            // the result is monotonic.
            let x = f64::max(w[0].as_float().abs(), w[1].as_float().abs());
            let largest = ((a.abs() * x + b.abs()) * x + c.abs()) * x + d.abs();
            if !(f1.is_finite() && f2.is_finite() && largest.is_finite()) {
                // no useful range: allow every output
                return vec![(0, std::u64::MAX)];
            }
            let slack = if a == 0.0 && b == 0.0 {
                0
            } else {
                ((8.0 * std::f64::EPSILON * largest).ceil() as u64).saturating_add(1)
            };

            let p1 = self.predict_to_int(&w[0]);
            let p2 = self.predict_to_int(&w[1]);
            ranges.push((u64::min(p1, p2).saturating_sub(slack),
                         u64::max(p1, p2).saturating_add(slack)));
        }
        return ranges;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::*;

    #[test]
    fn test_cubic() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2), (2, 3), (3, 8), (4, 20)]);

        let cubic_mod = CubicSplineModel::new(&md);

        assert_abs_diff_eq!(cubic_mod.predict_to_float(1.into()), 2.0, epsilon = 0.5);
        assert_abs_diff_eq!(cubic_mod.predict_to_float(4.into()), 20.0, epsilon = 0.5);
    }

    #[test]
    fn test_cubic2() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2), (2, 3), (3, 8), (4, 20), (5, 80)]);

        let cubic_mod = CubicSplineModel::new(&md);

        assert_abs_diff_eq!(cubic_mod.predict_to_float(1.into()), 2.0, epsilon = 0.5);
        assert_abs_diff_eq!(cubic_mod.predict_to_float(5.into()), 80.0, epsilon = 0.5);
    }

    #[test]
    fn test_cubic_dup() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2), (1, 2), (3, 8), (4, 20), (5, 80)]);

        let cubic_mod = CubicSplineModel::new(&md);

        assert_abs_diff_eq!(cubic_mod.predict_to_float(1.into()), 2.0, epsilon = 0.5);
        assert_abs_diff_eq!(cubic_mod.predict_to_float(5.into()), 80.0, epsilon = 0.5);
    }

    #[test]
    fn test_cubic_all_dup() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2), (1, 2), (1, 2)]);

        let cubic_mod = CubicSplineModel::new(&md);

        assert_abs_diff_eq!(cubic_mod.predict_to_float(1.into()), 2.0, epsilon = 0.5);
    }

    #[test]
    fn test_linear_spline_single() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 2)]);

        let cubic_mod = CubicSplineModel::new(&md);

        assert_eq!(cubic_mod.predict_to_int(1.into()), 2);
    }

    #[test]
    fn test_empty() {
        CubicSplineModel::new(&ModelData::empty());
    }

}
