// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::*;
use plr::regression::{GreedyPLR, OptimalPLR};

pub fn num_bits(largest_target: u64) -> u8 {
    let mut nbits = 0;
    while (1 << (nbits + 1)) - 1 <= largest_target {
        nbits += 1;
    }
    nbits -= 1;
    assert!((1 << (nbits + 1)) - 1 <= largest_target);

    return nbits;
}

pub fn common_prefix_size(data: &ModelData) -> u8 {
    let mut any_ones: u64 = 0;
    let mut no_ones: u64 = !0;

    for (x, _y) in data.iter_int_int() {
        any_ones |= x;
        no_ones &= x;
    }

    let any_zeros = !no_ones;

    let prefix_bits = any_zeros ^ any_ones;
    return (!prefix_bits).leading_zeros() as u8;
}

macro_rules! plr_with {
    ($plr: ty, $delta: ident, $data: ident) => {{
        let mut plr = <$plr>::new($delta);
        let mut segments = Vec::new();

        let mut last_x = -1.0;
        for (x, y) in $data.iter_float_float() {
            if x == last_x {
                continue;
            } else {
                last_x = x;
            }

            if let Some(seg) = plr.process(x, y) {
                assert!(!f64::is_nan(seg.slope));
                assert!(!f64::is_nan(seg.intercept));
                segments.push(seg);
            }
        }

        if let Some(seg) = plr.finish() {
            assert!(!f64::is_nan(seg.slope));
            assert!(!f64::is_nan(seg.intercept));
            segments.push(seg);
        }

        segments
    }};
}

pub fn plr(data: &ModelData, delta: f64, optimal: bool) -> (Vec<u64>, Vec<f64>) {
    let segments = if optimal {
        plr_with!(OptimalPLR, delta, data)
    } else {
        plr_with!(GreedyPLR, delta, data)
    };

    let points: Vec<u64> = segments.iter().map(|seg| seg.start as u64).collect();

    let coeffs = segments
        .iter()
        .flat_map(|seg| vec![seg.slope, seg.intercept])
        .collect();

    return (points, coeffs);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_prefix1() {
        let data = ModelData::IntKeyToIntPos(vec![(1, 0), (4, 4), (8, 8)]);

        assert_eq!(common_prefix_size(&data), 64 - 4);
    }

    #[test]
    fn test_common_prefix2() {
        let data = ModelData::IntKeyToIntPos(vec![(1, 0), (8, 1), (9, 4), (12, 8)]);

        assert_eq!(common_prefix_size(&data), 64 - 4);
    }
}
