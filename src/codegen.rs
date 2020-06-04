// < begin copyright > 
// Copyright Ryan Marcus 2020
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
use std::fmt;


enum StorageConf {
    Embed,
    Disk(String)
}

enum LayerParams {
    Constant(usize, Vec<ModelParam>),
    Array(usize, usize, Vec<ModelParam>),
    MixedArray(usize, usize, Vec<ModelParam>)
}

macro_rules! constant_name {
    ($layer:expr, $idx: expr) => {
        format!("L{}_PARAMETER{}", $layer, $idx)
    };
}


macro_rules! array_name {
    ($layer: expr) => {
        format!("L{}_PARAMETERS", $layer)
    }
}

impl LayerParams {

    fn new(idx: usize,
           array_access: bool,
           params_per_model: usize,
           params: Vec<ModelParam>) -> LayerParams {
        // first, if the underlying data is mixed, we can only support array mode.
        let first_param = params.first().unwrap();
        let mixed = !params.iter().all(|p| first_param.is_same_type(p));

        if mixed {
            return LayerParams::MixedArray(idx, params_per_model, params);
        }

        let param_size_bytes: usize = params.iter().map(|p| p.size()).sum();
        if array_access || param_size_bytes > 4096 {
            return LayerParams::Array(idx, params_per_model, params);
        }

        return LayerParams::Constant(idx, params);
    }
    
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

            LayerParams::Array(idx, _, params) => {
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
            },

            LayerParams::MixedArray(_, _, _) => {
                panic!("Cannot hardcode mixed array.");
            }
        };

        return Result::Ok(());
    }

    fn requires_malloc(&self) -> bool {
        return match self {
            LayerParams::Array(_, _, params) => {
                let array_size: usize = params.iter().map(|p| p.size()).sum();
                return array_size >= 4 * 1024;
            },
            LayerParams::MixedArray(_, _, _) => true,
            LayerParams::Constant(_, _) => false,
        }; 
    }

    fn pointer_type(&self) -> &'static str {
        assert!(self.requires_malloc());
        return match self {
            LayerParams::Array(_, _, params) => params[0].c_type(),
            LayerParams::MixedArray(_, _, _) => "char",
            LayerParams::Constant(_, _) => panic!("No pointer type for constant params")
        };
    }
    
    fn to_decl<T: Write>(&self, target: &mut T) -> Result<(), std::io::Error> {
        match self {
            LayerParams::Constant(_, _) => {
                panic!("Cannot forward-declare constants");
            }

            LayerParams::Array(idx, _, params) => {
                if !self.requires_malloc()  {
                    let num_items: usize = params.iter().map(|p| p.len()).sum();
                    writeln!(
                        target,
                        "{} {}[{}];",
                        params[0].c_type(),
                        array_name!(idx),
                        num_items
                    )?;
                } else { 
                    writeln!(
                        target,
                        "{}* {};",
                        params[0].c_type(),
                        array_name!(idx)
                    )?;
                }
            },

            LayerParams::MixedArray(idx, _, _) => {
                assert!(self.requires_malloc());
                writeln!(
                    target,
                    "char* {};",
                    array_name!(idx)
                )?;
            }
        };

        return Result::Ok(());
    }


    fn write_to<T: Write>(&self, target: &mut T) -> Result<(), std::io::Error> {
        match self {   
            LayerParams::Array(_idx, _, params) |
            LayerParams::MixedArray(_idx, _, params) => {
                let (first, rest) = params.split_first().unwrap();

                first.write_to(target)?;
                for itm in rest {
                    if let LayerParams::Array(_, _, _) = self {
                        assert!(first.is_same_type(itm));
                    }
                    itm.write_to(target)?;
                }
                return Ok(());
            },
            LayerParams::Constant(_, _) =>
                panic!("Cannot write constant parameters to binary file.")
        };
    }

    fn params(&self) -> &[ModelParam] {
        return match self {
            LayerParams::Array(_, _, params) |
            LayerParams::MixedArray(_, _, params)
                => params,
            LayerParams::Constant(_, params) => params
        };
    }

    fn index(&self) -> usize {
        return match self {
            LayerParams::Array(idx, _, _) |
            LayerParams::MixedArray(idx, _, _)
                => *idx,
            LayerParams::Constant(idx, _) => *idx
        };
    }

    fn params_per_model(&self) -> usize {
        return match self {
            LayerParams::Array(_idx, ppm, _params) |
            LayerParams::MixedArray(_idx, ppm, _params)
                => *ppm,
            LayerParams::Constant(_, params) => params.len()
        };
    }

    fn size(&self) -> usize {
        return self.params().iter().map(|p| p.size()).sum();
    }


    fn access_by_const<T: Write>(
        &self,
        target: &mut T,
        parameter_index: usize,
    ) -> Result<(), std::io::Error> {
        if let LayerParams::Constant(idx, _) = self {
            write!(target, "{}", constant_name!(idx, parameter_index))?;
            return Result::Ok(());
        }
        return self.access_by_ref(target, "0", parameter_index);
    }

    fn access_by_ref<T: Write>(
        &self,
        target: &mut T,
        model_index: &str,
        parameter_index: usize
    ) -> Result<(), std::io::Error> {

        if self.params()[0].is_array() {
            assert_eq!(self.params().len(), 1,
                       "Layer params with array had more than one member.");
            write!(target, "{}", array_name!(self.index()))?;
            return Result::Ok(());
        }
        
        match self {
            LayerParams::Constant(idx, _) => {
                panic!(
                    "Cannot access constant parameters by reference on layer {}",
                    idx
                );
            }

            LayerParams::Array(idx, params_per_model, params) => {
                if params[0].is_array() {
                    assert_eq!(params.len(), 1);
                }
                let expr = format!("{}*{} + {}",
                                   params_per_model, model_index, parameter_index);
                write!(target, "{}[{}]", array_name!(idx), expr)?;
            },

            LayerParams::MixedArray(idx, params_per_model, params) => {
                // determine the number of bytes for each model
                let mut bytes_per_model = 0;
                for item in params.iter().take(*params_per_model) {
                    bytes_per_model += item.size();
                }
                // determine the byte offset of this parameter
                let mut offset = 0;
                for item in params.iter().take(parameter_index) {
                    offset += item.size();
                }
                
                // we have to determine the type of the index being accessed
                // and add the appropiate cast.
                let c_type = params[parameter_index].c_type();
                let ptr_expr = format!("{} + ({} * {}) + {}",
                                       array_name!(idx),
                                       model_index, bytes_per_model,
                                       offset);
                                       
                write!(target, "*(({new_type}*) ({ptr_expr}))",
                       new_type=c_type, ptr_expr=ptr_expr)?;
                
            }
        };

        return Result::Ok(());
    }

    fn with_zipped_errors(&self, lle: Vec<u64>) -> LayerParams {
        
        let params = self.params();
        // integrate the errors into the model parameters of the last
        // layer to save a cache miss.
        
        // TODO we should add padding to make sure each of these are
        // cache-aligned. Also a lot of unneeded copying going on here...
        let combined_lle_params: Vec<ModelParam> =
            params.chunks(self.params_per_model())
            .zip(lle)
            .flat_map(|(mod_params, err)| {
                let mut to_r: Vec<ModelParam> = Vec::new();
                to_r.extend_from_slice(mod_params);
                to_r.push(ModelParam::Int(err));
                to_r
            }).collect();

        let is_constant = if let LayerParams::Constant(_, _) = self {
            true
        } else {
            false
        };
        
        return LayerParams::new(self.index(), is_constant, self.params_per_model() + 1,
                                combined_lle_params);
                                
    }
}

impl fmt::Display for LayerParams {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            LayerParams::Constant(idx, params) =>
                write!(f, "Constant(idx: {}, len: {}, malloc: {})",
                       idx, params.len(), self.requires_malloc()),
            LayerParams::Array(idx, ppm, params) =>
                write!(f, "Array(idx: {}, ppm: {}, len: {}, malloc: {})",
                       idx, ppm, params.len(), self.requires_malloc()),
            LayerParams::MixedArray(idx, ppm, params) =>
                write!(f, "MixedArray(idx: {}, ppm: {}, len: {}, malloc: {})",
                       idx, ppm, params.len(), self.requires_malloc())
                
        }
    }
}

fn params_for_layer(layer_idx: usize, models: &[Box<dyn Model>]) -> LayerParams {
    let params_per_model = models[0].params().len();
    let params = models.iter().flat_map(|m| m.params()).collect();
    return LayerParams::new(layer_idx,
                            models.len() > 1, // array access on non-singleton layers
                            params_per_model,
                            params);
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
            ModelDataType::Int128 => {
                if $needs_check {
                    format!("(i128pred > {0} - 1 ? {0} - 1 : i128pred)", $bound)
                } else {
                    format!("i128pred")
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
    data_dir: &str,
    build_time: u128,
    key_type: KeyType
) -> Result<(), std::io::Error> {
    // construct the code for the model parameters.
    let mut layer_params: Vec<LayerParams> = rmi
        .iter()
        .enumerate()
        .map(|(layer_idx, models)| params_for_layer(layer_idx, models))
        .collect();
    
    let report_last_layer_errors = last_layer_errors.is_some();

    let mut report_lle: Vec<u8> = Vec::new();
    if report_last_layer_errors {
        if let Some(lle) = last_layer_errors {
            assert!(!lle.is_empty());
            if lle.len() > 1 {
                let old_last = layer_params.pop().unwrap();
                let new_last = old_last.with_zipped_errors(lle);

                write!(report_lle, "  *err = ")?;
                new_last.access_by_ref(&mut report_lle, "modelIndex",
                                       new_last.params_per_model() - 1)?;
                writeln!(report_lle, ";")?;
                
                layer_params.push(new_last);

            } else {
                write!(report_lle, "  *err = {};", lle[0])?;
            }
        }
    }

    trace!("Layer parameters:");
    for lps in layer_params.iter() {
        trace!("{}", lps);
    }

    writeln!(data_output, "namespace {} {{", namespace)?;    
    
    let mut read_code = Vec::new();
    read_code.push("bool load(char const* dataPath) {".to_string());
            
    for lp in layer_params.iter() {
        match lp {
            // constants are put directly in the header 
            LayerParams::Constant(_idx, _) => lp.to_code(data_output)?,
            
            LayerParams::Array(idx, _, _) |
            LayerParams::MixedArray(idx, _, _) => {
                let data_path = Path::new(&data_dir)
                    .join(format!("{}_{}", namespace, array_name!(idx)));
                let f = File::create(data_path)
                    .expect("Could not write data file to RMI directory");
                let mut bw = BufWriter::new(f);
                
                lp.write_to(&mut bw)?; // write to data file
                lp.to_decl(data_output)?; // write to source code
                
                read_code.push("  {".to_string());
                read_code.push(format!("    std::ifstream infile(std::filesystem::path(dataPath) / \"{ns}_{fn}\", std::ios::in | std::ios::binary);",
                                       ns=namespace, fn=array_name!(idx)));
                read_code.push("    if (!infile.good()) return false;".to_string());
                if lp.requires_malloc() {
                    read_code.push(format!("    {} = ({}*) malloc({});",
                                           array_name!(idx), lp.pointer_type(), lp.size()));
                    read_code.push(format!("    if ({} == NULL) return false;",
                                           array_name!(idx)));
                }
                read_code.push(format!("    infile.read((char*){fn}, {size});",
                                       fn=array_name!(idx), size=lp.size()));
                read_code.push("    if (!infile.good()) return false;".to_string());
                read_code.push("  }".to_string());
            }
        }
    }
    read_code.push("  return true;".to_string());
    read_code.push("}".to_string());



    let mut free_code = Vec::new();
    free_code.push("void cleanup() {".to_string());
    // generate free code
    for lp in layer_params.iter() {
        if !lp.requires_malloc() { continue; }
        if let LayerParams::Array(idx, _, _) | LayerParams::MixedArray(idx, _, _) = lp {
            free_code.push(format!("    free({});", array_name!(idx)));
            continue;
        }
        panic!();
    }
    
    free_code.push("}".to_string());

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
    writeln!(code_output, "#include <cmath>")?;
    writeln!(code_output, "#include <fstream>")?;
    writeln!(code_output, "#include <filesystem>")?;
    writeln!(code_output, "#include <iostream>")?;

    writeln!(code_output, "namespace {} {{", namespace)?;

    for ln in read_code {
        writeln!(code_output, "{}", ln)?;
    }

    for ln in free_code {
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
        format!("uint64_t lookup({} key, size_t* err)", key_type.c_type())
    } else {
        format!("uint64_t lookup({} key)", key_type.c_type())
    };
    writeln!(code_output, "{} {{", lookup_sig)?;

    let mut needed_vars = HashSet::new();
    if rmi.len() > 1 {
        needed_vars.insert("size_t modelIndex;");
    }

    // determine if we have any layers with float (fpred) or int (ipred) outputs
    for layer in rmi.iter() {
        match layer[0].output_type() {
            ModelDataType::Int => needed_vars.insert("uint64_t ipred;"),
            ModelDataType::Float => needed_vars.insert("double fpred;"),
            ModelDataType::Int128 => needed_vars.insert("uint128_t i128pred;"),
        };
    }

    for var in needed_vars {
        writeln!(code_output, "  {}", var)?;
    }

    let model_size_bytes = rmi_size(&rmi, report_last_layer_errors);
    info!("Generated model size: {:?} ({} bytes)", ByteSize(model_size_bytes), model_size_bytes);

    let mut last_model_output = key_type.to_model_data_type();
    let mut needs_bounds_check = true;

    for (layer_idx, layer) in rmi.into_iter().enumerate() {
        let layer_param = &layer_params[layer_idx];
        let required_type = layer[0].input_type();

        let current_model_output = layer[0].output_type();

        let var_name = match current_model_output {
            ModelDataType::Int => "ipred",
            ModelDataType::Float => "fpred",
            ModelDataType::Int128 => "i128pred"
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
                layer_param.access_by_ref(code_output, "modelIndex", pidx)?;
                write!(code_output, ", ")?;
            }
        }
        writeln!(code_output, "({})key);", required_type.c_type())?;

        last_model_output = layer[0].output_type();
        needs_bounds_check = layer[0].needs_bounds_check();
    }

    writeln!(code_output, "{}", str::from_utf8(&report_lle).unwrap())?;

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

    writeln!(header_output, "bool load(char const* dataPath);")?;
    writeln!(header_output, "void cleanup();")?;

    
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
                  trained_model: TrainedRMI,
                  build_time: u128,
                  data_dir: &str,
                  key_type: KeyType,
                  include_errors: bool) -> Result<(), std::io::Error> {
    
    let f1 = File::create(format!("{}.cpp", namespace)).expect("Could not write RMI CPP file");
    let mut bw1 = BufWriter::new(f1);
    
    let f2 =
        File::create(format!("{}_data.h", namespace)).expect("Could not write RMI data file");
    let mut bw2 = BufWriter::new(f2);
    
    let f3 = File::create(format!("{}.h", namespace)).expect("Could not write RMI header file");
    let mut bw3 = BufWriter::new(f3);
    
    let lle = if include_errors {
        Some(trained_model.last_layer_max_l1s)
    } else {
        None
    };
        
    return generate_code(
        &mut bw1,
        &mut bw2,
        &mut bw3,
        namespace,
        trained_model.num_rows,
        trained_model.rmi,
        lle,
        data_dir,
        build_time,
        key_type
    );
        
    
}
