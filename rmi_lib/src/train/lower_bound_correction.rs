use crate::models::*;


fn find_first_below<T: Copy>(data: &[Option<T>], idx: usize) -> Option<(usize, T)> {
    assert!(idx < data.len());
    if idx == 0 { return None; }
    
    let mut i = idx - 1;
    loop {
        if let Some(v) = data[i] { return Some((i, v)); }
        if i == 0 { return None; }
        i -= 1;
    }
}

fn find_first_above<T: Copy>(data: &[Option<T>], idx: usize) -> Option<(usize, T)> {
    assert!(idx < data.len());
    if idx == data.len() - 1 { return None; }
        
    let mut i = idx + 1;
    loop {
        if let Some(v) = data[i] { return Some((i, v)); }
        if i == data.len() - 1 { return None; }
        i += 1;
    }
}

// next_for_leaf[i] stores the (key index, key) pairs for the first key in the 
// leaf model after leaf i. next_for_leaf[last leaf index] stores the maximum possible key.
fn compute_next_for_leaf<T: TrainingKey>(num_leaf_models: u64,
                                        num_keys: usize,
                                        first_key_for_leaf: &[Option<(usize, T)>])
                                        -> Vec<(usize, T)> {

    let mut next_for_leaf = vec![(0, T::zero_value()) ; num_leaf_models as usize];
    let mut idx: usize = 0;
    while idx < num_leaf_models as usize {
        match find_first_above(&first_key_for_leaf, idx as usize) {
            Some((next_leaf_idx, val)) => {
                assert!(next_leaf_idx > idx);
                for i in idx..next_leaf_idx {
                    next_for_leaf[i] = val;
                }
                idx = next_leaf_idx;
            },
            None => {
                for i in idx..num_leaf_models as usize {
                    next_for_leaf[i] = (num_keys, T::max_value());
                }
                break;
            }
        }
    }

    return next_for_leaf;
}

fn compute_prev_for_leaf<T: TrainingKey>(num_leaf_models: u64,
                                        last_key_for_leaf: &[Option<(usize, T)>])
                                        -> Vec<(usize, T)> {
    
    let mut prev_for_leaf: Vec<(usize, T)>
        = vec![(0, T::zero_value()) ; num_leaf_models as usize];
    let mut idx: usize = num_leaf_models as usize - 1;
    while idx > 0 {
        match find_first_below(&last_key_for_leaf, idx as usize) {
            Some((prev_leaf_idx, val)) => {
                assert!(prev_leaf_idx < idx);
                for i in prev_leaf_idx+1..idx+1 {
                    prev_for_leaf[i] = val;
                }
                idx = prev_leaf_idx;
            },
            None => { break; }
        }
    }

    return prev_for_leaf;
    
}


pub struct LowerBoundCorrection<T> {
    first: Vec<Option<(usize, T)>>,
    last: Vec<Option<(usize, T)>>,
    next: Vec<(usize, T)>,
    prev: Vec<(usize, T)>,
    run_lengths: Vec<u64>
}

impl <T: TrainingKey> LowerBoundCorrection<T> {
    pub fn new<F>(pred_func: F, num_leaf_models: u64, data: &RMITrainingData<T>) -> LowerBoundCorrection<T>
    where F: Fn(T) -> u64 {
    
        let mut first_key_for_leaf: Vec<Option<(usize, T)>>
            = vec![None ; num_leaf_models as usize];
        let mut last_key_for_leaf: Vec<Option<(usize, T)>>
            = vec![None ; num_leaf_models as usize];
        let mut max_run_length: Vec<u64> = vec![0 ; num_leaf_models as usize];
        
        let mut last_target = 0;
        let mut current_run_length = 0;
        let mut current_run_key = data.get_key(0);
        for (x, y) in data.iter() {
            let leaf_idx = pred_func(x.into());
            let target = u64::min(num_leaf_models - 1, leaf_idx) as usize;
            
            if target == last_target && x == current_run_key {
                current_run_length += 1;
            } else if target != last_target || x != current_run_key {
                max_run_length[last_target] = u64::max(
                    max_run_length[last_target],
                    current_run_length
                );
                
                current_run_length = 1;
                current_run_key = x;
                last_target = target;
            }
            
            if first_key_for_leaf[target].is_none() {
                first_key_for_leaf[target] = Some((y, x));
            }
            last_key_for_leaf[target] = Some((y, x));
        }

        // Flush the final run.  The loop above only commits a run to
        // `max_run_length` when it sees a transition, so the last run -- and the
        // only run in a single-run dataset -- would otherwise never be recorded,
        // making `longest_run` under-report (returning 0 for single-run data).
        max_run_length[last_target] = u64::max(
            max_run_length[last_target],
            current_run_length
        );

        let next_for_leaf = compute_next_for_leaf(num_leaf_models, data.len(), &first_key_for_leaf);
        let prev_for_leaf = compute_prev_for_leaf(num_leaf_models, &last_key_for_leaf);
        
        return LowerBoundCorrection {
            first: first_key_for_leaf,
            last: last_key_for_leaf,
            next: next_for_leaf,
            prev: prev_for_leaf,
            run_lengths: max_run_length
        };
    }
    
    pub fn first_key(&self, leaf_idx: usize) -> Option<T> {
        return self.first[leaf_idx].map(|x| x.1);
    }
    
    pub fn last_key(&self, leaf_idx: usize) -> Option<T> {
        return self.last[leaf_idx].map(|x| x.1);
    }

    pub fn next(&self, leaf_idx: usize) -> (usize, T) {
        return self.next[leaf_idx];
    }
    
    pub fn next_index(&self, leaf_idx: usize) -> usize {
        return self.next[leaf_idx].0;
    }
    
    pub fn prev_key(&self, leaf_idx: usize) -> T {
        return self.prev[leaf_idx].1;
    }

    pub fn longest_run(&self, leaf_idx: usize) -> u64 {
        return self.run_lengths[leaf_idx];
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Regression: a single-run dataset (every entry shares one key, so the
    // run-length loop never sees a transition) must still report the full run
    // length.  Before the final-run flush, `longest_run` returned 0 here,
    // because a run is only committed to `max_run_length` on a transition.
    //
    // All iterated entries fall in one run, so `longest_run(0)` must equal the
    // number of entries the trainer iterates.  We derive the expected count
    // from `data.iter()` rather than hard-coding it, since `FixDupsIter` adds a
    // trailing replay of the last element.
    #[test]
    fn single_run_reports_full_length() {
        let pairs: Vec<(u64, usize)> = vec![(5, 0), (5, 1), (5, 2)];
        let data = RMITrainingData::<u64>::new(Box::new(pairs));
        let expected = data.iter().count() as u64;
        assert!(expected > 0, "test setup: dataset must be non-empty");

        let lbc = LowerBoundCorrection::new(|k: u64| k >> 60, 16, &data);
        assert_eq!(
            lbc.longest_run(0),
            expected,
            "single-run dataset should report longest_run equal to its iterated length"
        );
    }
}
