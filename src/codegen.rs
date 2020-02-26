// < begin copyright >
// Copyright Ryan Marcus 2019
//
// See root directory of this project for license terms.
//
// < end copyright >
use crate::models::Model;
use crate::models::*;
use bytesize::ByteSize;
use log::*;
use std::collections::HashSet;
use std::io::Write;
use std::str;
use crate::train::TrainedRMI;
use std::fs::File;
use std::io::BufWriter;
use std::path::Path;


enum StorageConf {
    Embed,
    Disk(String)
}

enum LayerParams {
    Constant(usize, Vec<ModelParam>),
    Array(usize, Vec<ModelParam>),
}

macro_rules! constant_name {
    ($layer:expr, $idx: expr) => {
        format!("L{}_PARAMETER{}", $layer, $idx)
    };
}


// slight hack here -- special-case index 9999 to be the model
// error parameters (if you have a 9999 layer RMI, something else is wrong).
const MODEL_ERROR_MAGIC_NUM: usize = 9999;
macro_rules! array_name {
    ($layer: expr) => {
        if $layer == &MODEL_ERROR_MAGIC_NUM {
            format!("errors")
        } else {
            format!("L{}_PARAMETERS", $layer)
        }
    };
}

impl LayerParams {    
    fn to_code<T: Write>(&self, target: &mut T) -> Result<(), std::io::Error> {
        match self {
            LayerParams::Constant(idx, params) => {
                for (p_idx, param) in params.iter().enumerate() {
                    writeln!(
                        target,
                        "const {} {}{} = {};",
                        param.c_type(),
                        constant_name!(idx, p_idx),
                        param.c_type_mod(),
                        param.c_val()
                    )?;
                }
            }

            LayerParams::Array(idx, params) => {
                write!(
                    target,
                    "const {} {}[] = {{",
                    params[0].c_type(),
                    array_name!(idx)
                )?;

                let (last, rest) = params.split_last().unwrap();
                for param in rest {
                    write!(target, "{},", param.c_val())?;
                }
                write!(target, "{}", last.c_val())?;
                writeln!(target, "}};")?;
            }
        };

        return Result::Ok(());
    }

    fn to_decl<T: Write>(&self, target: &mut T) -> Result<(), std::io::Error> {
        match self {
            LayerParams::Constant(_, _) => {
                panic!("Cannot forward-declare constants");
            }

            LayerParams::Array(idx, params) => {
                let num_items: usize = params.iter().map(|p| p.len()).sum();
                writeln!(
                    target,
                    "{} {}[{}];",
                    params[0].c_type(),
                    array_name!(idx),
                    num_items
                )?;
            }
        };

        return Result::Ok(());
    }


    fn write_to<T: Write>(&self, target: &mut T) -> Result<(), std::io::Error> {
        if let LayerParams::Array(_idx, params) = self {
            let (first, rest) = params.split_first().unwrap();

            first.write_to(target)?;
            for itm in rest {
                assert!(first.is_same_type(itm));
                itm.write_to(target)?;
            }
            return Ok(());
        }
        panic!("Cannot write constant parameters to binary file.");
    }

    fn size(&self) -> usize {
        match self {
            LayerParams::Array(_, params) => params.iter().map(|p| p.size()).sum(),
            LayerParams::Constant(_, params) => params.iter().map(|p| p.size()).sum()
        }
    }

    fn access_by_const<T: Write>(
        &self,
        target: &mut T,
        parameter_index: usize,
    ) -> Result<(), std::io::Error> {
        match self {
            LayerParams::Constant(idx, _) => {
                write!(target, "{}", constant_name!(idx, parameter_index))?;
            }

            LayerParams::Array(idx, _) => {
                write!(target, "{}[{}]", array_name!(idx), parameter_index)?;
            }
        };

        return Result::Ok(());
    }

    fn access_by_ref<T: Write>(
        &self,
        target: &mut T,
        parameter_index: &str,
    ) -> Result<(), std::io::Error> {
        match self {
            LayerParams::Constant(idx, _) => {
                panic!(
                    "Cannot access constant parameters by reference on layer {}",
                    idx
                );
            }

            LayerParams::Array(idx, _) => {
                write!(target, "{}[{}]", array_name!(idx), parameter_index)?;
            }
        };

        return Result::Ok(());
    }
}

fn params_for_layer(layer_idx: usize, models: &[Box<dyn Model>]) -> LayerParams {
    if models.len() == 1 {
        // treat this data as constant, as long as it isn't too big.
        let params: Vec<ModelParam> = models[0].params();
        let size: usize = params.iter().map(|mp| mp.size()).sum();
        
        if size < 4096 {
            return LayerParams::Constant(layer_idx, params);
        } else {
            return LayerParams::Array(layer_idx, params);
        }
    }

    // we have more than one model in this layer, so we need to make an array.
    let params = models.iter().flat_map(|m| m.params()).collect();
    return LayerParams::Array(layer_idx, params);
}

macro_rules! model_index_from_output {
    ($from: expr, $bound: expr, $needs_check: expr) => {
        match $from {
            ModelDataType::Float => {
                if $needs_check {
                    format!("FCLAMP(fpred, {}.0 - 1.0)", $bound)
                } else {
                    format!("(uint64_t) fpred")
                }
            }
            ModelDataType::Int => {
                if $needs_check {
                    format!("(ipred > {0} - 1 ? {0} - 1 : ipred)", $bound)
                } else {
                    format!("ipred")
                }
            }
        }
    };
}

pub fn rmi_size(rmi: &[Vec<Box<dyn Model>>], report_last_layer_errors: bool) -> u64 {
    // compute the RMI size (used in the header, compute here before consuming)
    let mut num_total_bytes = 0;
    for layer in rmi.iter() {
        let model_on_this_layer_size: usize = layer[0].params().iter().map(|p| p.size()).sum();
        
        // assume all models on this layer have the same size
        num_total_bytes += model_on_this_layer_size * layer.len();
    }

    if report_last_layer_errors {
        num_total_bytes += rmi.last().unwrap().len() * 8;
    }
    
    return num_total_bytes as u64;
}

fn generate_code<T: Write>(
    code_output: &mut T,
    data_output: &mut T,
    header_output: &mut T,
    namespace: &str,
    total_rows: usize,
    rmi: Vec<Vec<Box<dyn Model>>>,
    last_layer_errors: Option<Vec<u64>>,
    storage: StorageConf,
    build_time: u128,
) -> Result<(), std::io::Error> {
    // construct the code for the model parameters.
    let mut layer_params: Vec<LayerParams> = rmi
        .iter()
        .enumerate()
        .map(|(layer_idx, models)| params_for_layer(layer_idx, models))
        .collect();

    let report_last_layer_errors = last_layer_errors.is_some();

    let mut report_lle = String::new();
    if report_last_layer_errors {
        if let Some(lle) = last_layer_errors {
            assert!(!lle.is_empty());
            if lle.len() > 1 {
                report_lle = String::from("  *err = errors[modelIndex];");
            } else {
                report_lle = format!("  *err = {};", lle[0]);
            }
            
            layer_params.push(LayerParams::Array(MODEL_ERROR_MAGIC_NUM,
                                                 vec![ModelParam::IntArray(lle)]));
        }
    }

    writeln!(data_output, "namespace {} {{", namespace)?;    
    
    let mut read_code = Vec::new();
    match &storage {
        // embed the data directly inside of the header files
        StorageConf::Embed => {
            for lp in layer_params.iter() {
                lp.to_code(data_output)?;
            }
        },

        // store the data on disk, add code to load it
        StorageConf::Disk(path) => {
            read_code.push(format!("void load(char const* dataPath) {{"));
            
            for lp in layer_params.iter() {
                match lp {
                    // constants are still put directly in the header
                    LayerParams::Constant(_, _) => lp.to_code(data_output)?,
                    
                    LayerParams::Array(idx, _) => {
                        let data_path = Path::new(&path).join(format!("{}_{}", namespace, array_name!(idx)));
                        let f = File::create(data_path).expect("Could not write data file");
                        let mut bw = BufWriter::new(f);

                        lp.write_to(&mut bw)?; // write to data file
                        lp.to_decl(data_output)?; // write to source code

                        read_code.push(format!("  {{"));
                        read_code.push(format!("    std::ifstream infile(std::filesystem::path(dataPath) / \"{ns}_{fn}\", std::ios::in | std::ios::binary);",
                                               ns=namespace, fn=array_name!(idx)));
                        read_code.push(format!("    infile.read((char*){fn}, {size});",
                                               fn=array_name!(idx), size=lp.size()));
                        read_code.push(format!("  }}"));
                    }
                }
            }
            read_code.push(format!("}}"));

        }
    };

    writeln!(data_output, "}} // namespace")?;

    // get all of the required stdlib function signatures together
    // TODO assumes all layers are homogenous
    let mut decls = HashSet::new();
    let mut sigs = HashSet::new();
    for layer in rmi.iter() {
        for stdlib in layer[0].standard_functions() {
            decls.insert(stdlib.decl().to_string());
            sigs.insert(stdlib.code().to_string());
        }
    }

    writeln!(code_output, "#include \"{}.h\"", namespace)?;
    writeln!(code_output, "#include \"{}_data.h\"", namespace)?;
    writeln!(code_output, "#include <math.h>")?;
    writeln!(code_output, "#include <fstream>")?;
    writeln!(code_output, "#include <filesystem>")?;
    writeln!(code_output, "#include <iostream>")?;

    writeln!(code_output, "namespace {} {{", namespace)?;

    for ln in read_code {
        writeln!(code_output, "{}", ln)?;
    }
    
    for decl in decls {
        writeln!(code_output, "{}", decl)?;
    }

    for sig in sigs {
        writeln!(code_output, "{}", sig)?;
    }

    // next, the model sigs
    sigs = HashSet::new();
    for layer in rmi.iter() {
        sigs.insert(layer[0].code());
    }

    for sig in sigs {
        writeln!(code_output, "{}", sig)?;
    }

    writeln!(
        code_output,
        "
inline size_t FCLAMP(double inp, double bound) {{
  if (inp < 0.0) return 0;
  return (inp > bound ? bound : (size_t)inp);
}}\n"
    )?;

    let lookup_sig = if report_last_layer_errors {
        "uint64_t lookup(uint64_t key, size_t* err)"
    } else {
        "uint64_t lookup(uint64_t key)"
    };
    writeln!(code_output, "{} {{", lookup_sig)?;

    // determine if we have any layers with float (fpred) or int (ipred) outputs
    let mut needed_vars = HashSet::new();
    if rmi.len() > 1 {
        needed_vars.insert("size_t modelIndex;");
    }

    for layer in rmi.iter() {
        match layer[0].output_type() {
            ModelDataType::Int => needed_vars.insert("uint64_t ipred;"),
            ModelDataType::Float => needed_vars.insert("double fpred;"),
        };
    }

    for var in needed_vars {
        writeln!(code_output, "  {}", var)?;
    }

    let model_size_bytes = rmi_size(&rmi, report_last_layer_errors);
    info!("Generated model size: {:?} ({} bytes)", ByteSize(model_size_bytes), model_size_bytes);

    let mut last_model_output = ModelDataType::Int;
    let mut needs_bounds_check = true;

    for (layer_idx, layer) in rmi.into_iter().enumerate() {
        let layer_param = &layer_params[layer_idx];
        let required_type = layer[0].input_type();

        let current_model_output = layer[0].output_type();

        let var_name = match current_model_output {
            ModelDataType::Int => "ipred",
            ModelDataType::Float => "fpred",
        };

        let num_parameters = layer[0].params().len();
        if layer.len() == 1 {
            // use constant indexing, only one model
            write!(
                code_output,
                "  {} = {}(",
                var_name,
                layer[0].function_name()
            )?;

            for pidx in 0..num_parameters {
                layer_param.access_by_const(code_output, pidx)?;
                write!(code_output, ", ")?;
            }
        } else {
            // we need to get the model index based on the previous
            // prediction, and then use ref accessing
            writeln!(
                code_output,
                "  modelIndex = {};",
                model_index_from_output!(last_model_output, layer.len(), needs_bounds_check)
            )?;

            write!(
                code_output,
                "  {} = {}(",
                var_name,
                layer[0].function_name()
            )?;

            for pidx in 0..num_parameters {
                let expr = format!("{}*modelIndex + {}", num_parameters, pidx);
                layer_param.access_by_ref(code_output, expr.as_str())?;
                write!(code_output, ", ")?;
            }
        }
        writeln!(code_output, "({})key);", required_type.c_type())?;

        last_model_output = layer[0].output_type();
        needs_bounds_check = layer[0].needs_bounds_check();
    }

    writeln!(code_output, "{}", report_lle)?;

    writeln!(
        code_output,
        "  return {};",
        model_index_from_output!(last_model_output, total_rows, true)
    )?; // always bounds check the last level
    writeln!(code_output, "}}")?;

    writeln!(code_output, "}} // namespace")?;

    // write out our forward declarations
    writeln!(header_output, "#include <cstddef>")?;
    writeln!(header_output, "#include <cstdint>")?;
    writeln!(header_output, "namespace {} {{", namespace)?;

    if let StorageConf::Disk(_) = storage {
        writeln!(header_output, "void load(char const* dataPath);")?;
    }

    
    if !report_last_layer_errors {
        writeln!(header_output, "#ifdef EXTERN_RMI_LOOKUP")?;
        writeln!(header_output, "extern \"C\" uint64_t lookup(uint64_t key);")?;
        writeln!(header_output, "#endif")?;
    }

    writeln!(
        header_output,
        "const size_t RMI_SIZE = {};",
        model_size_bytes
    )?;
    assert!(build_time <= u128::from(std::u64::MAX));
    writeln!(
        header_output,
        "const uint64_t BUILD_TIME_NS = {};",
        build_time
    )?;
    writeln!(header_output, "const char NAME[] = \"{}\";", namespace)?;
    writeln!(header_output, "{};", lookup_sig)?;
    writeln!(header_output, "}}")?;

    return Result::Ok(());
}


pub fn output_rmi(namespace: &str,
                  last_layer_errors: bool,
                  trained_model: TrainedRMI,
                  num_rows: usize,
                  build_time: u128,
                  data_dir: Option<&str>) -> Result<(), std::io::Error> {
    
    let f1 = File::create(format!("{}.cpp", namespace)).expect("Could not write RMI CPP file");
    let mut bw1 = BufWriter::new(f1);
    
    let f2 =
        File::create(format!("{}_data.h", namespace)).expect("Could not write RMI data file");
    let mut bw2 = BufWriter::new(f2);
    
    let f3 = File::create(format!("{}.h", namespace)).expect("Could not write RMI header file");
    let mut bw3 = BufWriter::new(f3);
    
    let lle = if last_layer_errors {
        Some(trained_model.last_layer_max_l1s)
    } else {
        None
    };

    let conf = match data_dir {
        None => StorageConf::Embed,
        Some(s) => StorageConf::Disk(String::from(s))
    };
    
    return generate_code(
        &mut bw1,
        &mut bw2,
        &mut bw3,
        namespace,
        num_rows,
        trained_model.rmi,
        lle,
        conf,
        build_time,
    );
        
    
}
