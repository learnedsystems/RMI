// < begin copyright > 
// Copyright Ryan Marcus 2020
// 
// See root directory of this project for license terms.
// 
// < end copyright > 
 
 
use memmap::MmapOptions;
use rmi_lib::{RMITrainingData, RMITrainingDataIteratorProvider, KeyType, ModelInput};
use byteorder::{LittleEndian, ReadBytesExt};
use std::fs::File;

pub enum DataType {
    UINT64,
    UINT32,
    FLOAT64
}

struct SliceAdapter {
    data: memmap::Mmap,
    dtype: DataType,
    length: usize
}

impl RMITrainingDataIteratorProvider for SliceAdapter {
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (ModelInput, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }
    
    fn get(&self, idx: usize) -> Option<(ModelInput, usize)> {
        if idx >= self.length { return None; };
        let mi: ModelInput = match self.dtype {
            DataType::UINT64 => {
                (&self.data[8 + idx * 8..8 + (idx + 1) * 8])
                    .read_u64::<LittleEndian>().unwrap().into()
            },
            DataType::UINT32 => {
                (&self.data[8 + idx * 4..8 + (idx + 1) * 4])
                    .read_u32::<LittleEndian>().unwrap().into()
            },
            DataType::FLOAT64 => {
                (&self.data[8 + idx * 8..8 + (idx + 1) * 8])
                    .read_f64::<LittleEndian>().unwrap().into()
            }
        };

        return Some((mi, idx));
    }
    fn key_type(&self) -> KeyType {
        match self.dtype {
            DataType::UINT64 => KeyType::U64,
            DataType::UINT32 => KeyType::U32,
            DataType::FLOAT64 => KeyType::F64
        }
    }
    fn len(&self) -> usize { self.length }
}


pub fn load_data(filepath: &str,
                 dt: DataType) -> (usize, RMITrainingData) {
    let fd = File::open(filepath).unwrap_or_else(|_| {
        panic!("Unable to open data file at {}", filepath)
    });

    let mmap = unsafe { MmapOptions::new().map(&fd).unwrap() };
    let num_items = (&mmap[0..8]).read_u64::<LittleEndian>().unwrap() as usize;

    let sa = SliceAdapter {
        data: mmap,
        dtype: dt,
        length: num_items
    };

    return (num_items, RMITrainingData::new(Box::new(sa)));
}
