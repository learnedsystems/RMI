use crate::models::KeyType;
use crate::models::ModelParam;
use crate::train::TrainedRMI;
use serde::{Deserialize, Serialize};
use std::fs::{self, File};
use std::io::Write;
use std::path::{Path, PathBuf};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ParamKind {
    Int,
    Float,
    ShortArray,
    IntArray,
    Int32Array,
    FloatArray,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", content = "value")]
pub enum ParamValue {
    Int(u64),
    Float(f64),
    ShortArray(Vec<u16>),
    IntArray(Vec<u64>),
    Int32Array(Vec<u32>),
    FloatArray(Vec<f64>),
}

impl From<&ModelParam> for ParamValue {
    fn from(param: &ModelParam) -> Self {
        match param {
            ModelParam::Int(v) => ParamValue::Int(*v),
            ModelParam::Float(v) => ParamValue::Float(*v),
            ModelParam::ShortArray(v) => ParamValue::ShortArray(v.clone()),
            ModelParam::IntArray(v) => ParamValue::IntArray(v.clone()),
            ModelParam::Int32Array(v) => ParamValue::Int32Array(v.clone()),
            ModelParam::FloatArray(v) => ParamValue::FloatArray(v.clone()),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParameterDescriptor {
    pub kind: ParamKind,
    pub len: usize,
}

impl From<&ModelParam> for ParameterDescriptor {
    fn from(param: &ModelParam) -> Self {
        let kind = match param {
            ModelParam::Int(_) => ParamKind::Int,
            ModelParam::Float(_) => ParamKind::Float,
            ModelParam::ShortArray(_) => ParamKind::ShortArray,
            ModelParam::IntArray(_) => ParamKind::IntArray,
            ModelParam::Int32Array(_) => ParamKind::Int32Array,
            ModelParam::FloatArray(_) => ParamKind::FloatArray,
        };

        ParameterDescriptor {
            kind,
            len: param.len(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerStorage {
    Constant { values: Vec<ParamValue> },
    Array { file: String },
    MixedArray { file: String },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerMetadata {
    pub index: usize,
    pub model_type: String,
    pub num_models: usize,
    pub params_per_model: usize,
    pub parameters: Vec<ParameterDescriptor>,
    pub storage: LayerStorage,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheFixMetadata {
    pub file: String,
    pub line_size: usize,
    pub points: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RmiMetadata {
    pub namespace: String,
    pub key_type: String,
    pub models: String,
    pub branching_factor: u64,
    pub build_time_ns: u128,
    pub num_rmi_rows: usize,
    pub num_data_rows: usize,
    pub model_avg_error: f64,
    pub model_avg_l2_error: f64,
    pub model_avg_log2_error: f64,
    pub model_max_error: u64,
    pub model_max_error_idx: usize,
    pub model_max_log2_error: f64,
    pub last_layer_reports_error: bool,
    pub layers: Vec<LayerMetadata>,
    pub cache_fix: Option<CacheFixMetadata>,
}

impl RmiMetadata {
    pub fn manifest_path<P: AsRef<Path>>(data_dir: P, namespace: &str) -> PathBuf {
        data_dir.as_ref().join(namespace).join("manifest.json")
    }
}

pub fn write_metadata<P: AsRef<Path>>(namespace: &str,
                                      data_dir: P,
                                      key_type: KeyType,
                                      rmi: &TrainedRMI,
                                      last_layer_reports_error: bool,
                                      layers: Vec<LayerMetadata>,
                                      cache_fix: Option<CacheFixMetadata>)
                                      -> std::io::Result<()> {

    let metadata = RmiMetadata {
        namespace: namespace.to_string(),
        key_type: key_type.as_str().to_string(),
        models: rmi.models.clone(),
        branching_factor: rmi.branching_factor,
        build_time_ns: rmi.build_time,
        num_rmi_rows: rmi.num_rmi_rows,
        num_data_rows: rmi.num_data_rows,
        model_avg_error: rmi.model_avg_error,
        model_avg_l2_error: rmi.model_avg_l2_error,
        model_avg_log2_error: rmi.model_avg_log2_error,
        model_max_error: rmi.model_max_error,
        model_max_error_idx: rmi.model_max_error_idx,
        model_max_log2_error: rmi.model_max_log2_error,
        last_layer_reports_error,
        layers,
        cache_fix,
    };

    let manifest_path = RmiMetadata::manifest_path(data_dir, namespace);
    if let Some(parent) = manifest_path.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut file = File::create(manifest_path)?;
    let serialized = serde_json::to_vec_pretty(&metadata)
        .expect("serialization to JSON should not fail");
    file.write_all(&serialized)?;
    Ok(())
}

