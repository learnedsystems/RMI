// < begin copyright > 
// Copyright Ryan Marcus 2020
// 
// See root directory of this project for license terms.
// 
// < end copyright > 
 
 

use crate::models::RMITrainingData;
use byteorder::{LittleEndian, ReadBytesExt};
use log::debug;
use std::fs::File;
use std::io::BufReader;

pub enum DataType {
    UINT64,
    UINT32,
    FLOAT64
}

pub fn load_data(filepath: &str,
                 dt: DataType) -> (usize, RMITrainingData) {
    let fd = File::open(filepath).unwrap_or_else(|_| {
        panic!("Unable to open data file at {}", filepath)
    });
    
    let mut reader = BufReader::new(fd);

    let num_items = reader.read_u64::<LittleEndian>().unwrap() as usize;

    return match dt {
        DataType::UINT32 => {
            let mut keys = vec![0; num_items];
            reader.read_u32_into::<LittleEndian>(&mut keys).unwrap();
            process_u64(keys.into_iter().map(u64::from).collect())
        },
        DataType::UINT64 => {
            let mut keys = vec![0; num_items];
            reader.read_u64_into::<LittleEndian>(&mut keys).unwrap();
            process_u64(keys)
        },
        DataType::FLOAT64 => {
            let mut keys = vec![0.0; num_items];
            reader.read_f64_into::<LittleEndian>(&mut keys).unwrap();
            process_f64(keys)
        }
    };
}

fn process_u64(keys: Vec<u64>) -> (usize, RMITrainingData) {
    let has_duplicates = has_duplicates(&keys);
    let mut init_data: Vec<(u64, usize)> = Vec::with_capacity(keys.len());

    for (idx, k) in keys.into_iter().enumerate() {
        init_data.push((k, idx));
    }

    if has_duplicates {
        resolve_dup_keys(&mut init_data);
    }

    let orig_size = init_data.len();
    return (orig_size, RMITrainingData::new(Box::new(init_data)));
}
    
fn process_f64(keys: Vec<f64>) -> (usize, RMITrainingData) {
    // assume float datasets have no (meaningful) duplicate keys
    let init_data: Vec<(f64, usize)> = keys.into_iter()
        .enumerate()
        .map(|(idx, k)| (k, idx))
        .collect();

    return (init_data.len(), RMITrainingData::new(Box::new(init_data)));
}

fn has_duplicates<T: PartialEq>(data: &[T]) -> bool {
    let mut dup_count = 0;
    for i in 0..data.len() - 1 {
        if data[i] == data[i + 1] {
            dup_count += 1;
        }
    }

    if dup_count > 0 {
        debug!("Duplicate keys in dataset: {}", dup_count);
        return true;
    }

    return false;
}

fn resolve_dup_keys(data: &mut [(u64, usize)]) {
    if data.len() <= 1 {
        return;
    }

    for idx in 0..data.len() - 1 {
        let (x1, y1) = data[idx];
        let (x2, _y2) = data[idx + 1];

        if x1 == x2 {
            data[idx + 1] = (x2, y1);
        }
    }
}
