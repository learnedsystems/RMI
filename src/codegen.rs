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

enum LayerParams {
    Constant(usize, Vec<ModelParam>),
    Array(usize, Vec<ModelParam>),
}

macro_rules! constant_name {
    ($layer:expr, $idx: expr) => {
        format!("L{}_PARAMETER{}", $layer, $idx)
    };
}

macro_rules! array_name {
    ($layer: expr) => {
        format!("L{}_PARAMETERS", $layer);
    };
}

impl LayerParams {
    fn to_code<T: Write>(&self, target: &mut T) -> Result<(), std::io::Error> {
        match self {
            LayerParams::Constant(idx, params) => {
                for (p_idx, param) in params.iter().enumerate() {
                    writeln!(
                        target,
                        "const static {} {}{} = {};",
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
                    "const static {} {}[] = {{",
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
        let params: Vec<ModelParam> = models[0].params();
        return LayerParams::Constant(layer_idx, params);
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

pub fn generate_code<T: Write>(
    code_output: &mut T,
    data_output: &mut T,
    header_output: &mut T,
    namespace: String,
    total_rows: usize,
    rmi: Vec<Vec<Box<dyn Model>>>,
    last_layer_errors: Option<Vec<u64>>,
    build_time: u128,
) -> Result<(), std::io::Error> {
    // construct the code for the model parameters.
    let layer_params: Vec<LayerParams> = rmi
        .iter()
        .enumerate()
        .map(|(layer_idx, models)| params_for_layer(layer_idx, models))
        .collect();

    let report_last_layer_errors = last_layer_errors.is_some();

    writeln!(data_output, "namespace {} {{", namespace)?;
    for lp in layer_params.iter() {
        lp.to_code(data_output)?;
    }

    if let Some(lle) = last_layer_errors.as_ref() {
        if lle.len() > 1 {
            // save the last layer error data if there is more than one model
            // on the final layer.
            write!(data_output, "const size_t errors[] = {{")?;
            let (last, rest) = lle.split_last().unwrap();
            for err in rest {
                write!(data_output, "{},", err)?;
            }
            writeln!(data_output, "{} }};", last)?;
        }
    }

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
    writeln!(code_output, "namespace {} {{", namespace)?;
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

    // compute the RMI size (used in the header, compute here before consuming)
    let model_size_bytes = {
        let mut num_total_params = 0;
        for layer in rmi.iter() {
            let model_on_this_layer_size: usize = layer[0].params().iter().map(|p| p.size()).sum();

            // assume all models on this layer have the same size
            num_total_params += model_on_this_layer_size * layer.len();
        }

        if report_last_layer_errors {
            num_total_params += rmi.last().unwrap().len();
        }
        num_total_params as u64 * 8
    };

    info!("Generated model size: {:?}", ByteSize(model_size_bytes));

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

    if report_last_layer_errors {
        if let Some(lle) = last_layer_errors {
            assert!(!lle.is_empty());
            if lle.len() > 1 {
                writeln!(code_output, "  *err = errors[modelIndex];")?;
            } else {
                writeln!(code_output, "  *err = {};", lle[0])?;
            }
        }
    }

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
