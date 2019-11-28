// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::ModelData;
use byteorder::{LittleEndian, ReadBytesExt};
use log::debug;
use log::*;
use std::fs::File;
use std::io::BufReader;

pub enum DataType {
    UINT64,
    UINT32,
}

pub fn load_data(filepath: &str, dt: DataType, downsample: usize) -> (usize, ModelData) {
    let fd = File::open(filepath).unwrap();
    let mut reader = BufReader::new(fd);

    let num_items = reader.read_u64::<LittleEndian>().unwrap() as usize;

    let keys = match dt {
        DataType::UINT32 => {
            let mut keys = vec![0; num_items];
            reader.read_u32_into::<LittleEndian>(&mut keys).unwrap();
            keys.into_iter().map(u64::from).collect()
        }
        DataType::UINT64 => {
            let mut keys = vec![0; num_items];
            reader.read_u64_into::<LittleEndian>(&mut keys).unwrap();
            keys
        }
    };

    let has_duplicates = has_duplicates(&keys);
    let mut init_data: Vec<(u64, u64)> = Vec::with_capacity(keys.len());

    for (idx, k) in keys.into_iter().enumerate() {
        init_data.push((k, idx as u64));
    }

    if has_duplicates {
        resolve_dup_keys(&mut init_data);
    }

    let orig_size = init_data.len();

    // it is critical that we downsample AFTER assigning indexes and
    // not before.
    if downsample > 1 {
        trace!("Downsampling by a factor of {}", downsample);
        init_data = init_data.into_iter().step_by(downsample).collect();
        info!("Downsampled from {} to {}", orig_size, init_data.len());
    }

    return (orig_size, ModelData::IntKeyToIntPos(init_data));
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

fn resolve_dup_keys(data: &mut [(u64, u64)]) {
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
