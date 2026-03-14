mod codegen;
mod binary;
mod models;
mod train;
mod cache_fix;
mod manifest;

pub mod optimizer;
pub use models::{RMITrainingData, RMITrainingDataIteratorProvider, ModelInput};
pub use models::KeyType;
pub use optimizer::find_pareto_efficient_configs;
pub use train::{train, train_for_size, train_bounded};
pub use codegen::rmi_size;
pub use codegen::output_rmi;
pub use manifest::{CacheFixMetadata, LayerMetadata, ParameterDescriptor, ParamKind, ParamValue, RmiMetadata};
pub use binary::{Model, Stage, RMIModel};
