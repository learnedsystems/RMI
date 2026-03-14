// < begin copyright >
// Copyright Ryan Marcus 2020
//
// See root directory of this project for license terms.
//
// < end copyright >

use crate::models::*;
use log::trace;
use std::convert::TryFrom;

const DEFAULT_EPSILON: usize = 1;

#[derive(Clone, Copy, Debug)]
struct Point {
    x: f64,
    y: i64,
}

#[derive(Clone, Copy, Debug)]
struct Slope {
    dx: f64,
    dy: i64,
}

impl Slope {
    fn from_points(a: Point, b: Point) -> Self {
        Self {
            dx: a.x - b.x,
            dy: a.y - b.y,
        }
    }

    fn lt(self, other: Self) -> bool {
        (self.dy as f64) * other.dx < self.dx * (other.dy as f64)
    }

    fn gt(self, other: Self) -> bool {
        (self.dy as f64) * other.dx > self.dx * (other.dy as f64)
    }

    fn eq(self, other: Self) -> bool {
        ((self.dy as f64) * other.dx - self.dx * (other.dy as f64)).abs() <= f64::EPSILON
    }

    fn as_f64(self) -> f64 {
        (self.dy as f64) / self.dx
    }
}

#[derive(Clone, Debug)]
struct CanonicalSegment {
    rectangle: [Point; 4],
    first_x: f64,
}

impl CanonicalSegment {
    fn new_one_point(p0: Point, p1: Point, first_x: f64) -> Self {
        Self {
            rectangle: [p0, p1, p0, p1],
            first_x,
        }
    }

    fn new(rectangle: [Point; 4], first_x: f64) -> Self {
        Self { rectangle, first_x }
    }

    fn get_first_x(&self) -> f64 {
        self.first_x
    }

    fn one_point(&self) -> bool {
        self.rectangle[0].x == self.rectangle[2].x
            && self.rectangle[0].y == self.rectangle[2].y
            && self.rectangle[1].x == self.rectangle[3].x
            && self.rectangle[1].y == self.rectangle[3].y
    }

    fn get_slope_range(&self) -> (f64, f64) {
        if self.one_point() {
            return (0.0, 1.0);
        }

        let min = Slope::from_points(self.rectangle[2], self.rectangle[0]).as_f64();
        let max = Slope::from_points(self.rectangle[3], self.rectangle[1]).as_f64();
        (min, max)
    }

    fn get_intersection(&self) -> (f64, f64) {
        let p0 = self.rectangle[0];
        let p1 = self.rectangle[1];
        let p2 = self.rectangle[2];
        let p3 = self.rectangle[3];
        let slope1 = Slope::from_points(p2, p0);
        let slope2 = Slope::from_points(p3, p1);

        if self.one_point() || slope1.eq(slope2) {
            return (p0.x, p0.y as f64);
        }

        let p0p1 = Slope::from_points(p1, p0);
        let a = slope1.dx * (slope2.dy as f64) - (slope1.dy as f64) * slope2.dx;
        let b = (p0p1.dx * (slope2.dy as f64) - (p0p1.dy as f64) * slope2.dx) / a;

        let ix = p0.x + b * slope1.dx;
        let iy = p0.y as f64 + b * (slope1.dy as f64);
        (ix, iy)
    }

    fn get_floating_point_segment(&self, origin: f64) -> (f64, f64) {
        if self.one_point() {
            return (
                0.0,
                (self.rectangle[0].y + self.rectangle[1].y) as f64 / 2.0,
            );
        }

        let (ix, iy) = self.get_intersection();
        let (min_slope, max_slope) = self.get_slope_range();
        let slope = (min_slope + max_slope) / 2.0;
        let intercept = iy - (ix - origin) * slope;
        (slope, intercept)
    }
}

#[derive(Debug)]
struct OptimalPiecewiseLinearModel {
    epsilon: i64,
    lower: Vec<Point>,
    upper: Vec<Point>,
    first_x: f64,
    lower_start: usize,
    upper_start: usize,
    points_in_hull: usize,
    rectangle: [Point; 4],
}

impl OptimalPiecewiseLinearModel {
    fn new(epsilon: usize) -> Self {
        let epsilon = i64::try_from(epsilon).expect("epsilon too large for i64");
        Self {
            epsilon,
            lower: Vec::with_capacity(1 << 12),
            upper: Vec::with_capacity(1 << 12),
            first_x: 0.0,
            lower_start: 0,
            upper_start: 0,
            points_in_hull: 0,
            rectangle: [Point { x: 0.0, y: 0 }; 4],
        }
    }

    fn cross(o: Point, a: Point, b: Point) -> f64 {
        let oa = Slope::from_points(a, o);
        let ob = Slope::from_points(b, o);
        oa.dx * (ob.dy as f64) - (oa.dy as f64) * ob.dx
    }

    fn add_point(&mut self, x: f64, y: usize) -> bool {
        let y = i64::try_from(y).expect("position exceeds i64 range");
        let p1 = Point {
            x,
            y: y.saturating_add(self.epsilon),
        };
        let p2 = Point {
            x,
            y: y.saturating_sub(self.epsilon),
        };

        if self.points_in_hull == 0 {
            self.first_x = x;
            self.rectangle[0] = p1;
            self.rectangle[1] = p2;
            self.upper.clear();
            self.lower.clear();
            self.upper.push(p1);
            self.lower.push(p2);
            self.upper_start = 0;
            self.lower_start = 0;
            self.points_in_hull += 1;
            return true;
        }

        if self.points_in_hull == 1 {
            self.rectangle[2] = p2;
            self.rectangle[3] = p1;
            self.upper.push(p1);
            self.lower.push(p2);
            self.points_in_hull += 1;
            return true;
        }

        let slope1 = Slope::from_points(self.rectangle[2], self.rectangle[0]);
        let slope2 = Slope::from_points(self.rectangle[3], self.rectangle[1]);

        let outside_line1 = Slope::from_points(p1, self.rectangle[2]).lt(slope1);
        let outside_line2 = Slope::from_points(p2, self.rectangle[3]).gt(slope2);

        if outside_line1 || outside_line2 {
            self.points_in_hull = 0;
            return false;
        }

        if Slope::from_points(p1, self.rectangle[1]).lt(slope2) {
            let mut min = Slope::from_points(self.lower[self.lower_start], p1);
            let mut min_i = self.lower_start;

            for i in (self.lower_start + 1)..self.lower.len() {
                let val = Slope::from_points(self.lower[i], p1);
                if val.gt(min) {
                    break;
                }
                min = val;
                min_i = i;
            }

            self.rectangle[1] = self.lower[min_i];
            self.rectangle[3] = p1;
            self.lower_start = min_i;

            let mut end = self.upper.len();
            while end >= self.upper_start + 2
                && Self::cross(self.upper[end - 2], self.upper[end - 1], p1) <= 0.0
            {
                end -= 1;
            }

            self.upper.truncate(end);
            self.upper.push(p1);
        }

        if Slope::from_points(p2, self.rectangle[0]).gt(slope1) {
            let mut max = Slope::from_points(self.upper[self.upper_start], p2);
            let mut max_i = self.upper_start;

            for i in (self.upper_start + 1)..self.upper.len() {
                let val = Slope::from_points(self.upper[i], p2);
                if val.lt(max) {
                    break;
                }
                max = val;
                max_i = i;
            }

            self.rectangle[0] = self.upper[max_i];
            self.rectangle[2] = p2;
            self.upper_start = max_i;

            let mut end = self.lower.len();
            while end >= self.lower_start + 2
                && Self::cross(self.lower[end - 2], self.lower[end - 1], p2) >= 0.0
            {
                end -= 1;
            }

            self.lower.truncate(end);
            self.lower.push(p2);
        }

        self.points_in_hull += 1;
        true
    }

    fn get_segment(&self) -> CanonicalSegment {
        if self.points_in_hull == 1 {
            return CanonicalSegment::new_one_point(
                self.rectangle[0],
                self.rectangle[1],
                self.first_x,
            );
        }

        CanonicalSegment::new(self.rectangle, self.first_x)
    }
}

fn maybe_add_gap_point<T: TrainingKey, F: FnMut(T, usize)>(curr: T, next: T, y: usize, mut add: F) {
    let plus = curr.plus_epsilon();
    if plus.as_float() < next.as_float() {
        add(plus, y);
    }
}

fn make_segmentation<T: TrainingKey>(
    data: &RMITrainingData<T>,
    start: usize,
    end: usize,
    epsilon: usize,
) -> Vec<CanonicalSegment> {
    let mut segments = Vec::new();
    let mut pla = OptimalPiecewiseLinearModel::new(epsilon);

    let mut add_point = |x: T, y: usize| {
        if !pla.add_point(x.as_float(), y) {
            segments.push(pla.get_segment());
            let accepted = pla.add_point(x.as_float(), y);
            debug_assert!(accepted, "fresh segment must accept first point");
        }
    };

    add_point(data.get_key(start), start);

    for i in (start + 1)..(end.saturating_sub(1)) {
        let curr = data.get_key(i);
        let prev = data.get_key(i - 1);
        if curr == prev {
            let next = data.get_key(i + 1);
            maybe_add_gap_point(curr, next, i, |x, y| add_point(x, y));
        } else {
            add_point(curr, i);
        }
    }

    if end >= start + 2 && data.get_key(end - 1) != data.get_key(end - 2) {
        add_point(data.get_key(end - 1), end - 1);
    }

    if end == data.len() {
        add_point(data.get_key(data.len() - 1).plus_epsilon(), data.len());
    }

    segments.push(pla.get_segment());
    segments
}

fn build_segments<T: TrainingKey>(
    data: &RMITrainingData<T>,
    epsilon: usize,
) -> (Vec<f64>, Vec<f64>, Vec<u64>) {
    if data.len() == 0 {
        return (vec![0.0], vec![0.0], vec![u64::MAX]);
    }

    let canonical = make_segmentation(data, 0, data.len(), epsilon);

    let mut intercepts = Vec::with_capacity(canonical.len());
    let mut slopes = Vec::with_capacity(canonical.len());
    let mut boundaries = Vec::with_capacity(canonical.len());

    for (idx, segment) in canonical.iter().enumerate() {
        let (slope, intercept) = segment.get_floating_point_segment(0.0);
        intercepts.push(intercept);
        slopes.push(slope);

        let boundary = if idx + 1 < canonical.len() {
            canonical[idx + 1].get_first_x() as u64
        } else {
            u64::MAX
        };

        boundaries.push(boundary);
        trace!(
            "PLA segment {} slope={} intercept={} boundary={}",
            idx,
            slope,
            intercept,
            boundary
        );
    }

    (intercepts, slopes, boundaries)
}

pub struct OptimalPLAModel {
    epsilon: usize,
    intercepts: Vec<f64>,
    slopes: Vec<f64>,
    boundaries: Vec<u64>,
}

impl OptimalPLAModel {
    pub fn new<T: TrainingKey>(data: &RMITrainingData<T>) -> OptimalPLAModel {
        Self::with_epsilon(data, DEFAULT_EPSILON)
        
    }

    pub fn with_epsilon<T: TrainingKey>(
        data: &RMITrainingData<T>,
        epsilon: usize,
    ) -> OptimalPLAModel {
        // eprintln!("Building Optimal PLA model with epsilon={}", epsilon);
        let (intercepts, slopes, boundaries) = build_segments(data, epsilon);
        OptimalPLAModel {
            epsilon,
            intercepts,
            slopes,
            boundaries,
        }
    }
}

impl Model for OptimalPLAModel {
    fn predict_to_float(&self, inp: &ModelInput) -> f64 {
        if self.boundaries.is_empty() {
            return 0.0;
        }

        let key = inp.as_int();
        let idx = match self.boundaries.binary_search(&key) {
            Ok(exact) => exact + 1,
            Err(insert) => insert,
        };

        let bounded_idx = idx.min(self.intercepts.len() - 1);
        self.slopes[bounded_idx].mul_add(inp.as_float(), self.intercepts[bounded_idx])
    }

    fn input_type(&self) -> ModelDataType {
        ModelDataType::Float
    }
    fn output_type(&self) -> ModelDataType {
        ModelDataType::Float
    }

    fn params(&self) -> Vec<ModelParam> {
        vec![
            ModelParam::Int(self.intercepts.len() as u64),
            ModelParam::FloatArray(self.intercepts.clone()),
            ModelParam::FloatArray(self.slopes.clone()),
            ModelParam::IntArray(self.boundaries.clone()),
        ]
    }

    fn code(&self) -> String {
        String::from(
            "inline double optimal_pla(uint64_t length, const double intercepts[], const double slopes[], const uint64_t boundaries[], double inp) {\n    uint64_t idx = bs_upper_bound(boundaries, length, (uint64_t)inp);\n    if (idx >= length) { idx = length - 1; }\n    return std::fma(slopes[idx], inp, intercepts[idx]);\n}",
        )
    }

    fn standard_functions(&self) -> HashSet<StdFunctions> {
        let mut to_r = HashSet::new();
        to_r.insert(StdFunctions::BinarySearch);
        to_r
    }

    fn function_name(&self) -> String {
        String::from("optimal_pla")
    }

    fn model_name(&self) -> &'static str {
        "optimal_pla"
    }

    fn needs_bounds_check(&self) -> bool {
        false
    }

    fn set_to_constant_model(&mut self, constant: u64) -> bool {
        self.epsilon = DEFAULT_EPSILON;
        self.intercepts = vec![constant as f64];
        self.slopes = vec![0.0];
        self.boundaries = vec![u64::MAX];
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fits_linear_run() {
        let mut pts = Vec::new();
        for i in 0..50u64 {
            pts.push((i, i * 2));
        }
        let md = ModelData::IntKeyToIntPos(pts);
        let pla = OptimalPLAModel::new(&md);

        assert_eq!(pla.boundaries.len(), 1);
        assert_eq!(pla.predict_to_int(10.into()), 20);
        assert_eq!(pla.predict_to_int(49.into()), 98);
    }

    #[test]
    fn makes_multiple_segments_when_needed() {
        let md = ModelData::IntKeyToIntPos(vec![
            (0, 0),
            (1, 1),
            (2, 2),
            (100, 120),
            (101, 121),
            (102, 122),
        ]);

        let pla = OptimalPLAModel::new(&md);
        assert!(pla.boundaries.len() >= 2);

        assert_eq!(pla.predict_to_int(0.into()), 0);
        assert_eq!(pla.predict_to_int(101.into()), 121);
    }

    #[test]
    fn empty_defaults_to_zero() {
        let md: ModelData<u64> = ModelData::empty();
        let pla = OptimalPLAModel::new(&md);
        assert_eq!(pla.predict_to_int(5.into()), 0);
    }

    #[test]
    fn handles_duplicate_runs() {
        let md = ModelData::IntKeyToIntPos(vec![(1, 0), (1, 1), (1, 2), (5, 3), (6, 4)]);
        let pla = OptimalPLAModel::new(&md);

        assert!(pla.boundaries.len() >= 1);
        assert_eq!(pla.predict_to_int(1.into()), 2);
    }
}
