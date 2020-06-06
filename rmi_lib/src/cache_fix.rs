use log::*;
use crate::RMITrainingData;
use crate::models::ModelInput;

#[derive(Debug)]
pub struct Spline {
    from_x: u64,
    from_y: usize,
    to_x: u64,
    to_y: usize
}

impl Spline {
    fn from(pt1: (ModelInput, usize), pt2: (ModelInput, usize)) -> Spline {
        assert!(pt1.0.as_int() <= pt2.0.as_int(),
                "Cannot construct spline from {:?} to {:?}", pt1, pt2);
        assert!(pt1.1 <= pt2.1,
                "Cannot construct spline from {:?} to {:?}", pt1, pt2);
        return Spline { from_x: pt1.0.as_int(), from_y: pt1.1,
                        to_x: pt2.0.as_int(), to_y: pt2.1 };
    }

    fn with_new_dest(&self, dest: (ModelInput, usize)) -> Spline {
        assert!(dest.0.as_int() >= self.from_x,
                "When source x is {}, cannot set dest x to {}",
                self.from_x, dest.0.as_int());
        assert!(dest.1 >= self.from_y);
        return Spline { from_x: self.from_x, from_y: self.from_y,
                        to_x: dest.0.as_int(), to_y: dest.1 };
    }

    fn start(&self) -> (ModelInput, usize) {
        return (self.from_x.into(), self.from_y);
    }
    
    fn end(&self) -> (ModelInput, usize) {
        return (self.to_x.into(), self.to_y);
    }

    fn predict(&self, inp: ModelInput) -> usize {
        let v0 = self.from_y as f64;
        let v1 = self.to_y as f64;
        let t = ((inp.as_int() - self.from_x) as f64) / (self.to_x - self.from_x) as f64;

        return (1.0 - t).mul_add(v0, t * v1) as usize;
    }
}

struct SplineFit {
    splines: Vec<Spline>,
    curr_pts: Vec<(ModelInput, usize)>,
    line_size: usize
}

impl SplineFit {

    pub fn new(line_size: usize) -> SplineFit {
        return SplineFit {
            splines: vec![],
            curr_pts: Vec::new(),
            line_size
        };
    }

    pub fn add_point(&mut self, point: (ModelInput, usize)) {
        if self.splines.is_empty() {
            self.splines.push(Spline::from(point, point));
            return;
        }
        // check to see if the current spline can include this point
        let last_spline = self.splines.last().unwrap();
        let proposed_spline = last_spline.with_new_dest(point);

        self.curr_pts.push(last_spline.end());
        if self.check_spline(&proposed_spline) {
            // accept this proposal, it works.
            let last_idx = self.splines.len() - 1;
            self.splines[last_idx] = proposed_spline;
        } else {
            // reject this proposal, start a new spline.
            let prev_pt = last_spline.end();
            assert!(point.0 > prev_pt.0,
                    "new point: {:?} prev point: {:?}",
                    point, prev_pt);
            debug_assert!(self.check_spline(&last_spline));
            self.splines.push(Spline::from(prev_pt, point));
            self.curr_pts.clear();
            self.curr_pts.push(point);
        }
    }


    pub fn finish(self) -> Vec<(ModelInput, usize)> {
        let mut to_r = Vec::with_capacity(self.splines.len() + 1);
        to_r.push(self.splines[0].start());

        for spline in self.splines {
            assert_eq!(*to_r.last().unwrap(), spline.start());
            to_r.push(spline.end());
        }
        
        return to_r;
    }

    /*fn predict(&self, key: ModelInput) {
        let lb_idx = self.splines.lower_bound_by(|x| x.to_x.cmp(&key.as_int()));
        info!("Found spline @ {} ({:?}) for key {}",
              lb_idx, self.splines[lb_idx], key.as_int());

        let pred = self.splines[lb_idx].predict(key);
        info!("Pred: {}", pred);

        let start_of_line = (pred / 8) * 8;
        info!("Line {} starts at: {}", pred / 8, start_of_line);
    }*/
    
    fn check_spline(&self, spline: &Spline) -> bool {
        for pt in self.curr_pts.iter() {
            let predicted_line: usize = spline.predict(pt.0) / self.line_size;
            let correct_line: usize = pt.1 / self.line_size;
            
            if predicted_line != correct_line {
                return false;
            }
        }

        return true;
    }
}

pub fn cache_fix(data: &RMITrainingData, line_size: usize) -> Vec<(ModelInput, usize)> {
    assert!(data.len() > line_size,
            "Cannot apply a cachefix with fewer items than the line size");
    info!("Fitting cachefix spline to {} datapoints", data.len());
    
    let mut fit = SplineFit::new(line_size);

    // Potential speedup here by carefully building a spline over the first
    // and last element of each cache line. Requires careful handling of duplicates,
    // especially when they cross cache lines.
    let mut last_key = 0.into();
    for (key, offset) in data.iter_unique() {
        assert!(key.minus_epsilon() >= last_key,
                "key: {:?} last key: {:?}, key - e: {:?}",
                key, last_key, key.minus_epsilon());
        if key.minus_epsilon() != last_key {
            fit.add_point((key.minus_epsilon(), offset));
        }
        
        fit.add_point((key, offset));
        last_key = key;
    }


    let spline = fit.finish();

    // ensure the spline points are monotonic if in debug mode
    #[cfg(debug_assertions)]
    {
        let mut last_x: ModelInput = 0.into();
        for (x, _y) in spline.iter() {
            debug_assert!(*x >= last_x,
                          "Spline model was non-monotonic!");
            last_x = *x;
        }
        trace!("Spline model was monotonic.");
    }

    
    info!("Bounded spline compressed data to {}% of original ({} points, constructed from {} points).",
          ((spline.len() as f64 / data.len() as f64)*100.0).round(),
          spline.len(), data.len());

    
    return spline;
}
