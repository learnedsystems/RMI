mod codegen;
mod models;
mod train;

pub mod optimizer;
pub use models::RMITrainingData;
pub use models::KeyType;
pub use optimizer::find_pareto_efficient_configs;
pub use train::train;
pub use codegen::rmi_size;
pub use codegen::output_rmi;
