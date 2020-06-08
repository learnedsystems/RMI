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
use std::convert::TryInto;

pub enum DataType {
    UINT64,
    UINT32,
    FLOAT64
}

struct SliceAdapterU64 {
    data: memmap::Mmap,
    length: usize
}

impl RMITrainingDataIteratorProvider for SliceAdapterU64 {
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (ModelInput, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }
    
    fn get(&self, idx: usize) -> Option<(ModelInput, usize)> {
        if idx >= self.length { return None; };
        let mi = u64::from_le_bytes((&self.data[8 + idx * 8..8 + (idx + 1) * 8])
                                    .try_into().unwrap());
        return Some((mi.into(), idx));
    }
    
    fn key_type(&self) -> KeyType {
        KeyType::U64
    }
    
    fn len(&self) -> usize { self.length }
}


struct SliceAdapterU32 {
    data: memmap::Mmap,
    length: usize
}

impl RMITrainingDataIteratorProvider for SliceAdapterU32 {
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (ModelInput, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }
    
    fn get(&self, idx: usize) -> Option<(ModelInput, usize)> {
        if idx >= self.length { return None; };
        let mi = (&self.data[8 + idx * 4..8 + (idx + 1) * 4])
            .read_u32::<LittleEndian>().unwrap().into();
        return Some((mi, idx));
    }
    
    fn key_type(&self) -> KeyType {
        KeyType::U32
    }
    
    fn len(&self) -> usize { self.length }
}

struct SliceAdapterF64 {
    data: memmap::Mmap,
    length: usize
}

impl RMITrainingDataIteratorProvider for SliceAdapterF64 {
    fn cdf_iter(&self) -> Box<dyn Iterator<Item = (ModelInput, usize)> + '_> {
        Box::new((0..self.length).map(move |i| self.get(i).unwrap()))
    }
    
    fn get(&self, idx: usize) -> Option<(ModelInput, usize)> {
        if idx >= self.length { return None; };
        let mi = (&self.data[8 + idx * 8..8 + (idx + 1) * 8])
            .read_f64::<LittleEndian>().unwrap().into();
        return Some((mi, idx));
    }
    
    fn key_type(&self) -> KeyType {
        KeyType::F64
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

    let rtd = match dt {
        DataType::UINT64 =>
            RMITrainingData::new(Box::new(
                SliceAdapterU64 { data: mmap, length: num_items }
            )),
        DataType::UINT32 =>
            RMITrainingData::new(Box::new(
                SliceAdapterU32 { data: mmap, length: num_items }
            )),
        DataType::FLOAT64 =>
            RMITrainingData::new(Box::new(
                SliceAdapterF64 { data: mmap, length: num_items }
            ))
    };

    return (num_items, rtd);
}
