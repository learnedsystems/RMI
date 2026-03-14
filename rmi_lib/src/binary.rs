use crate::models::{KeyType, ModelDataType, ModelParam};
use crate::train::TrainedRMI;

#[derive(Debug, Clone)]
pub enum BinaryParamKind {
    Int,
    Float,
    ShortArray,
    IntArray,
    Int32Array,
    FloatArray,
}

#[derive(Debug, Clone)]
pub struct BinaryModelParam {
    pub kind: BinaryParamKind,
    pub len: usize,
    pub value: ModelParam,
}

impl From<ModelParam> for BinaryModelParam {
    fn from(param: ModelParam) -> Self {
        let kind = match &param {
            ModelParam::Int(_) => BinaryParamKind::Int,
            ModelParam::Float(_) => BinaryParamKind::Float,
            ModelParam::ShortArray(_) => BinaryParamKind::ShortArray,
            ModelParam::IntArray(_) => BinaryParamKind::IntArray,
            ModelParam::Int32Array(_) => BinaryParamKind::Int32Array,
            ModelParam::FloatArray(_) => BinaryParamKind::FloatArray,
        };

        BinaryModelParam {
            kind,
            len: param.len(),
            value: param,
        }
    }
}

#[derive(Debug, Clone)]
pub struct Model {
    pub model_type: String,
    pub input_type: ModelDataType,
    pub output_type: ModelDataType,
    pub params: Vec<BinaryModelParam>,
    pub error: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct Stage {
    pub models: Vec<Model>,
}

#[derive(Debug, Clone)]
pub struct CacheFix {
    pub line_size: usize,
    pub spline_points: Vec<(u64, usize)>,
}

#[derive(Debug, Clone)]
pub struct RMIModel {
    
    pub branching_factor: u64,
    pub models: String,
    pub last_layer_reports_error: bool,
    pub num_rmi_rows: usize,
    pub num_data_rows: usize,
    pub model_avg_error: f64,
    pub model_avg_l2_error: f64,
    pub model_avg_log2_error: f64,
    pub model_max_error: u64,
    pub model_max_error_idx: usize,
    pub model_max_log2_error: f64,
    pub build_time: u128,
    pub last_layer_max_l1s: Vec<u64>,
    pub stages: Vec<Stage>,
    pub cache_fix: Option<CacheFix>,
}

impl RMIModel {
    pub fn from_trained(
        rmi: &TrainedRMI,
        key_type: KeyType,
        last_layer_reports_error: bool,
    ) -> RMIModel {
        let mut stages = Vec::new();

        for stage in &rmi.rmi {
            let mut models = Vec::new();
            for m in stage {
                let params = m.params().into_iter().map(Into::into).collect();

                models.push(Model {
                    model_type: m.model_name().to_string(),
                    input_type: m.input_type(),
                    output_type: m.output_type(),
                    params,
                    error: m.error_bound(),
                });
            }
            stages.push(Stage { models });
        }

        let cache_fix = rmi.cache_fix.as_ref().map(|(line_size, points)| CacheFix {
            line_size: *line_size,
            spline_points: points.clone(),
        });

        RMIModel {
            branching_factor: rmi.branching_factor,
            models: rmi.models.clone(),
            last_layer_reports_error,
            num_rmi_rows: rmi.num_rmi_rows,
            num_data_rows: rmi.num_data_rows,
            model_avg_error: rmi.model_avg_error,
            model_avg_l2_error: rmi.model_avg_l2_error,
            model_avg_log2_error: rmi.model_avg_log2_error,
            model_max_error: rmi.model_max_error,
            model_max_error_idx: rmi.model_max_error_idx,
            model_max_log2_error: rmi.model_max_log2_error,
            build_time: rmi.build_time,
            last_layer_max_l1s: rmi.last_layer_max_l1s.clone(),
            stages,
            cache_fix,
        }
    }
}

use std::io::{Result, Write};

fn write_string<W: Write>(w: &mut W, s: &str) -> Result<()> {
    let bytes = s.as_bytes();
    w.write_all(&(bytes.len() as u64).to_le_bytes())?;
    w.write_all(bytes)
}

fn write_model_data_type<W: Write>(w: &mut W, t: &ModelDataType) -> Result<()> {
    let code = match t {
        ModelDataType::Int => 0u8,
        ModelDataType::Int128 => 1u8,
        ModelDataType::Float => 2u8,
    };

    w.write_all(&[code])
}

fn write_param<W: Write>(w: &mut W, p: &BinaryModelParam) -> Result<()> {
    let kind_code = match p.kind {
        BinaryParamKind::Int => 0u8,
        BinaryParamKind::Float => 1u8,
        BinaryParamKind::ShortArray => 2u8,
        BinaryParamKind::IntArray => 3u8,
        BinaryParamKind::Int32Array => 4u8,
        BinaryParamKind::FloatArray => 5u8,
    };

    w.write_all(&[kind_code])?;
    w.write_all(&(p.len as u64).to_le_bytes())?;

    match &p.value {
        ModelParam::Int(v) => w.write_all(&v.to_le_bytes()),
        ModelParam::Float(v) => w.write_all(&v.to_le_bytes()),
        ModelParam::ShortArray(arr) => {
            for v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
        ModelParam::IntArray(arr) => {
            for v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
        ModelParam::Int32Array(arr) => {
            for v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
        ModelParam::FloatArray(arr) => {
            for v in arr {
                w.write_all(&v.to_le_bytes())?;
            }
            Ok(())
        }
    }
}

fn write_cache_fix<W: Write>(w: &mut W, cf: &CacheFix) -> Result<()> {
    w.write_all(&(cf.line_size as u64).to_le_bytes())?;
    w.write_all(&(cf.spline_points.len() as u64).to_le_bytes())?;
    for (key, offset) in &cf.spline_points {
        w.write_all(&key.to_le_bytes())?;
        w.write_all(&(*offset as u64).to_le_bytes())?;
    }
    Ok(())
}

fn write_key_type<W: Write>(w: &mut W, k: KeyType) -> Result<()> {
    let code = match k {
        KeyType::U32 => 0u8,
        KeyType::U64 => 1u8,
        KeyType::F64 => 2u8,
        KeyType::U128 => 3u8,
    };

    w.write_all(&[code])
}

impl RMIModel {
    pub fn save_binary(&self, path: &str) -> Result<()> {
        let mut f = std::fs::File::create(path)?;

        f.write_all(b"RMIB")?;
        f.write_all(&1u32.to_le_bytes())?;

        f.write_all(&self.branching_factor.to_le_bytes())?;
        f.write_all(&(self.num_rmi_rows as u64).to_le_bytes())?;
        f.write_all(&(self.num_data_rows as u64).to_le_bytes())?;
        f.write_all(&self.build_time.to_le_bytes())?;
        f.write_all(&self.model_avg_error.to_le_bytes())?;
        f.write_all(&self.model_avg_l2_error.to_le_bytes())?;
        f.write_all(&self.model_avg_log2_error.to_le_bytes())?;
        f.write_all(&self.model_max_error.to_le_bytes())?;
        f.write_all(&(self.model_max_error_idx as u64).to_le_bytes())?;
        f.write_all(&self.model_max_log2_error.to_le_bytes())?;
        f.write_all(&[self.last_layer_reports_error as u8])?;

        write_string(&mut f, &self.models)?;

        f.write_all(&(self.last_layer_max_l1s.len() as u64).to_le_bytes())?;
        for v in &self.last_layer_max_l1s {
            f.write_all(&v.to_le_bytes())?;
        }

        if let Some(cf) = &self.cache_fix {
            f.write_all(&[1u8])?;
            write_cache_fix(&mut f, cf)?;
        } else {
            f.write_all(&[0u8])?;
        }

        let stage_count = self.stages.len() as u64;
        f.write_all(&stage_count.to_le_bytes())?;

        for stage in &self.stages {
            let model_count = stage.models.len() as u64;
            f.write_all(&model_count.to_le_bytes())?;

            for m in &stage.models {
                write_string(&mut f, &m.model_type)?;
                write_model_data_type(&mut f, &m.input_type)?;
                write_model_data_type(&mut f, &m.output_type)?;

                match m.error {
                    Some(err) => {
                        f.write_all(&[1u8])?;
                        f.write_all(&err.to_le_bytes())?;
                    }
                    None => f.write_all(&[0u8])?,
                };

                f.write_all(&(m.params.len() as u64).to_le_bytes())?;
                for p in &m.params {
                    write_param(&mut f, p)?;
                }
            }
        }

        Ok(())
    }
}
