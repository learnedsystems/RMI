// < begin copyright > 
// Copyright Ryan Marcus 2020
// 
// See root directory of this project for license terms.
// 
// < end copyright > 
 
use crate::models::*;
use plr::regression::{OptimalPLR, GreedyPLR};
use superslice::*;
use log::*;


pub fn num_bits(largest_target: u64) -> u8 {
    let mut nbits = 0;
    while (1 << (nbits+1)) - 1 <= largest_target {
        nbits += 1;
    }
    nbits -= 1;
    assert!((1 << (nbits+1)) - 1 <= largest_target);

    return nbits;
}

pub fn common_prefix_size<T: TrainingKey>(data: &RMITrainingData<T>) -> u8 {
    let mut any_ones: u64 = 0;
    let mut no_ones: u64 = !0;

    for (x, _y) in data.iter_model_input() {
        any_ones |= x.as_int();
        no_ones &= x.as_int();
    }

    let any_zeros = !no_ones;

    let prefix_bits = any_zeros ^ any_ones;
    return (!prefix_bits).leading_zeros() as u8;        
}

fn common_prefix_size2(data: &[u64]) -> u8 {
    let mut any_ones: u64 = 0;
    let mut no_ones: u64 = !0;

    for x in data {
        any_ones |= x;
        no_ones &= x;
    }

    let any_zeros = !no_ones;

    let prefix_bits = any_zeros ^ any_ones;
    return (!prefix_bits).leading_zeros() as u8;        
}



pub fn radix_index(points: &[u64], num_bits: u8) -> Vec<u64> {
    // build the radix index
    let cps = common_prefix_size2(points);
    if cps != 0 {
        warn!("Radix index currently assumes the common prefix size is 0, but it was {}",
              cps);
    }
    
    let mut radix_index: Vec<u64> = vec![0 ; 1 << num_bits];

    let mut last_radix = 0;
    for (idx, p) in points.iter().enumerate() {
        let radix = p >> (64 - num_bits);
        assert!(radix < radix_index.len() as u64);

        if radix == last_radix { continue; }
        
        for i in last_radix+1..radix {
            radix_index[i as usize] = idx as u64; //radix_index[last_radix as usize] + 1;
        }
        radix_index[radix as usize] = idx as u64;
        last_radix = radix;
    }

    for i in last_radix+1..radix_index.len() as u64 {
        radix_index[i as usize] = points.len() as u64;
    }
    
    // end point
    radix_index.push(points.len() as u64);
    
    // verify the radix construction
    for p in points {
        let radix = p >> (64 - num_bits);
        let radix_lb = radix_index[radix as usize];
        let radix_ub = radix_index[radix as usize + 1];

        let correct_idx = (points.upper_bound(p) - 1) as u64;
        assert!(radix_lb <= correct_idx,
                "On key {} with radix {}, correct index was {}, but radix LB = {} and UB = {}",
                p, radix, correct_idx, radix_lb, radix_ub);
        assert!(radix_ub > correct_idx,
                "On key {} with radix {}, correct index was {}, but radix LB = {} and UB = {}",
                p, radix, correct_idx, radix_lb, radix_ub);
    }

    return radix_index;
}

macro_rules! plr_with {
    ($plr: ty, $delta: ident, $data: ident) => {{
        let mut plr = <$plr>::new($delta);
        let mut segments = Vec::new();

        let mut last_x = -1.0;
        for (inp, y) in $data.iter() {
            let x = inp.to_model_input().as_float();
            if (x - last_x).abs() < std::f64::EPSILON {
                continue;
            } else {
                last_x = x;
            }
            
            if let Some(seg) = plr.process(x, y as f64) {
                assert!(! f64::is_nan(seg.slope));
                assert!(! f64::is_nan(seg.intercept));
                segments.push(seg);
            }            
        }
        
        if let Some(seg) = plr.finish() {
            assert!(! f64::is_nan(seg.slope));
            assert!(! f64::is_nan(seg.intercept));
            segments.push(seg);
        }

        segments
    }}
}

pub fn plr<T: TrainingKey>(data: &RMITrainingData<T>,
                          delta: f64,
                          optimal: bool) -> (Vec<u64>, Vec<f64>) {
    let segments = if optimal {
        plr_with!(OptimalPLR, delta, data)
    } else {
        plr_with!(GreedyPLR, delta, data)
    };

    let mut points: Vec<u64> = segments.iter()
        .map(|seg| seg.start as u64)
        .collect();

    points[0] = u64::min(points[0], data.iter().next().unwrap().0.to_model_input().as_int());
    
    let coeffs = segments.iter()
        .flat_map(|seg| vec![seg.slope, seg.intercept])
        .collect();

    
    return (points, coeffs);
}


#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_common_prefix1() {
        let data = ModelData::IntKeyToIntPos(vec![
            (1, 0), (4, 4), (8, 8)
        ]);

        assert_eq!(common_prefix_size(&data), 64-4);
    }

    #[test]
    fn test_common_prefix2() {
        let data = ModelData::IntKeyToIntPos(vec![
            (1, 0), (8, 1), (9, 4), (12, 8)
        ]);

        assert_eq!(common_prefix_size(&data), 64-4);
    }
}
