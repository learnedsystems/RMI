// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::utils::{common_prefix_size, num_bits};
use crate::models::*;
use log::*;

pub struct RadixModel {
    params: (u8, u8),
}

impl RadixModel {
    pub fn new(data: &ModelData) -> RadixModel {
        if data.len() == 0 {
            return RadixModel { params: (0, 0) };
        }

        let largest_value = data.iter_int_int().map(|(_x, y)| y).max().unwrap();
        let bits = num_bits(largest_value);
        trace!(
            "Radix layer using {} bits, from largest value {} (max layers: {})",
            bits,
            largest_value,
            (1 << (bits + 1)) - 1
        );

        let common_prefix = common_prefix_size(data);
        trace!("Radix layer common prefix: {}", common_prefix);

        return RadixModel {
            params: (common_prefix, bits),
        };
    }
}

impl Model for RadixModel {
    fn predict_to_int(&self, inp: ModelInput) -> u64 {
        let (left_shift, num_bits) = self.params;

        let as_int: u64 = inp.as_int();
        let res = (as_int << left_shift) >> (64 - num_bits);

        return res;
    }

    fn input_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }
    fn output_type(&self) -> ModelDataType {
        return ModelDataType::Int;
    }

    fn params(&self) -> Vec<ModelParam> {
        return vec![self.params.0.into(), self.params.1.into()];
    }

    fn code(&self) -> String {
        return String::from(
            "
inline uint64_t radix(uint64_t prefix_length, uint64_t bits, uint64_t inp) {
    return (inp << prefix_length) >> (64 - bits);
}",
        );
    }

    fn function_name(&self) -> String {
        return String::from("radix");
    }
    fn needs_bounds_check(&self) -> bool {
        return false;
    }
    fn restriction(&self) -> ModelRestriction {
        return ModelRestriction::MustBeTop;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty() {
        RadixModel::new(&ModelData::empty());
    }

}
